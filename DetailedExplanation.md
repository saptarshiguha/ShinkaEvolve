# ShinkaEvolve: From Paper to Code Implementation

**A Comprehensive Guide to Understanding Open-Ended and Sample-Efficient Program Evolution**

---

## Table of Contents
1. [The Big Idea](#the-big-idea)
2. [Detailed Concepts](#detailed-concepts)
3. [Code Implementation](#code-implementation)
4. [AIME Case Study: End-to-End Example](#aime-case-study-end-to-end-example)

---

## The Big Idea

### What is ShinkaEvolve?

ShinkaEvolve is an **evolutionary framework that uses Large Language Models (LLMs) to automatically improve and optimize programs**. Think of it as "natural selection for code" - the system maintains a population of program variants, evaluates their fitness, and uses LLMs to generate improved versions through mutation and crossover.

### Why is it Needed?

Traditional code optimization methods struggle with:
- **Sample inefficiency**: Requiring thousands of evaluations to find good solutions
- **Closed exploration**: Getting stuck in local optima without discovering novel approaches
- **Manual tuning**: Needing human intervention to guide the search

### What Does ShinkaEvolve Achieve?

ShinkaEvolve introduces **three key algorithmic innovations** that dramatically improve sample efficiency while maintaining open-ended exploration:

1. **Adaptive Parent Sampling**: Intelligently balances exploring new approaches vs. exploiting known good solutions
2. **Code Novelty Rejection Sampling**: Prevents wasting evaluations on near-duplicate programs
3. **Bandit-Based LLM Ensemble Selection**: Dynamically chooses which LLM to use for maximum improvement

**Result**: State-of-the-art solutions with ~150 evaluations (vs. thousands in prior work)

---

## Detailed Concepts

### 1. Adaptive Parent Sampling

**Paper Concept**: Programs are selected as "parents" for mutation through a hierarchical strategy that balances exploration and exploitation.

#### How It Works:

**Power Law Sampling** (`database/parents.py:11-42`):
- Programs are ranked by performance
- Probability of selection follows P(rank) ∝ rank^(-α)
- α controls exploration/exploitation:
  - α = 0: uniform random (pure exploration)
  - α > 0: better programs selected more often (exploitation)
  - α = 1.0 (typical): top program selected ~31% of time

**Weighted Tree Selection** (`database/parents.py:274-455`):
- Uses archive of elite programs
- Each program gets weight: w_i = sigmoid(performance) × novelty_bonus
- Sigmoid scaling: s_i = σ(λ × (score_i - median_score) / MAD)
  - Makes selection robust to different problem scales
  - Prevents extreme scores from dominating
- Novelty bonus: h_i = 1/(1 + num_children)
  - Favors programs with fewer children (unexplored branches)

**Island-Based Evolution** (`database/islands.py`):
- Maintains 4+ separate populations (islands)
- Each island evolves independently
- Periodic migration between islands every N generations
- Preserves diversity, prevents premature convergence

#### Code Implementation:

```python
# shinka/database/parents.py:274-455
class WeightedSamplingStrategy(ParentSamplingStrategy):
    def sample_parent(self) -> Any:
        # Calculate baseline (median performance)
        alpha_0 = np.median(scores)

        # Robust scaling using Median Absolute Deviation
        mad = np.median([abs(score - alpha_0) for score in scores])
        scale_factor = max(mad, 1e-6)

        for i, program in enumerate(eligible_programs):
            # Sigmoid-scaled performance
            normalized_diff = (program.score - alpha_0) / scale_factor
            s_i = stable_sigmoid(lambda_ * normalized_diff)

            # Novelty bonus (fewer children = higher weight)
            h_i = 1 / (1 + program.children_count)

            # Combined weight
            w_i = s_i * h_i
            weights.append(w_i)

        # Sample parent based on normalized weights
        selected_parent = np.random.choice(programs, p=weights/sum(weights))
```

**Why This Matters**: Without adaptive sampling, evolution either gets stuck exploiting one good solution (local optimum) or wastes time exploring random variants. This strikes the perfect balance.

---

### 2. Code Novelty Rejection Sampling

**Paper Concept**: An embedding-based approach that filters redundant program variants, preventing near-duplicate mutations from consuming computational resources.

#### How It Works:

**Code Embedding** (`core/runner.py:1130-1166`):
- Each program is converted to a vector representation
- Uses `text-embedding-3-small` (OpenAI) or similar
- Only mutable code sections are embedded (immutable parts redacted)
- Embeddings stored in database for efficient similarity computation

**Similarity Checking** (`core/novelty_judge.py:60-172`):
- When a new program is generated, compute cosine similarity with existing programs in the same island
- If max_similarity > threshold (e.g., 0.95): program is potentially redundant
- **Rejection Sampling Loop**:
  1. Attempt 1: Sample new parent/inspirations, generate different mutation
  2. Attempt 2: Another resample if still too similar
  3. Attempt 3: Final attempt
  4. If all fail: reject the program entirely

**LLM-Based Novelty Verification** (`core/novelty_judge.py:174-221`):
- When similarity is high but uncertain, query LLM as judge
- LLM compares proposed code vs. most similar existing code
- Answers: "NOVEL" or "NOT NOVEL" with explanation
- Catches semantic novelty that embeddings miss

#### Code Implementation:

```python
# shinka/core/novelty_judge.py:60-172
class NoveltyJudge:
    def assess_novelty_with_rejection_sampling(
        self, exec_fname, code_embedding, parent_program, database
    ) -> Tuple[bool, dict]:

        for attempt in range(self.max_novelty_attempts):
            # Compute similarities with programs in same island
            similarity_scores = database.compute_similarity(
                code_embedding, parent_program.island_idx
            )

            max_similarity = max(similarity_scores)

            if max_similarity <= self.similarity_threshold:
                # Low similarity - accept immediately
                return True, novelty_metadata

            # High similarity - check with LLM if configured
            if self.novelty_llm_client is not None:
                most_similar_program = database.get_most_similar_program(
                    code_embedding, parent_program.island_idx
                )

                is_novel, explanation, cost = self.check_llm_novelty(
                    proposed_code, most_similar_program
                )

                if is_novel:
                    return True, novelty_metadata  # Accept despite high similarity

            # Reject and retry with different parent/inspirations
            continue

        # All attempts exhausted - reject program
        return False, novelty_metadata
```

**Why This Matters**: In early experiments, ~40% of generated programs were near-duplicates. Novelty rejection saves massive computational resources by avoiding redundant evaluations.

---

### 3. Bandit-Based LLM Ensemble Selection

**Paper Concept**: A UCB1-inspired adaptive strategy that dynamically prioritizes which LLMs contribute most effectively to fitness improvements throughout evolution.

#### How It Works:

**Multi-Armed Bandit Formulation**:
- Each LLM is an "arm" to pull
- Reward: improvement over parent's score
- Goal: maximize cumulative improvement over evolution

**Asymmetric UCB Algorithm** (`llm/dynamic_sampling.py:135-559`):

```
score_i = exploitation_i + exploration_i

exploitation_i = exp(mean_improvement_i - best_observed) / (best - worst)
exploration_i = c × sqrt(2 × log(total_pulls) / pulls_i)
```

**Key Features**:
- **Asymmetric Scaling**: Only counts positive improvements (failures don't penalize as much)
- **Exponential Scaling**: Uses log-space arithmetic for numerical stability
- **Adaptive Normalization**: Scales rewards based on observed range (robust to different problems)
- **ε-Greedy**: With probability ε, explores random LLM (default ε=0.2)

**Reward Computation**:
- Baseline: max(parent_score, initial_score)
- Reward: child_score - baseline
- If child fails: reward = -∞ (in log space) or 0 (asymmetric)
- Decay factor (α=0.95): older observations count less

#### Code Implementation:

```python
# shinka/llm/dynamic_sampling.py:135-366
class AsymmetricUCB(BanditBase):
    def update(self, arm, reward, baseline):
        # Compute improvement over baseline
        r = reward - baseline if reward else self._impute_worst_reward()

        # Asymmetric: only count positive improvements
        if self.asymmetric_scaling:
            r = max(r, 0.0)

        # Exponential scaling for stability
        if self.use_exponential_scaling:
            contrib_log = np.log(np.expm1(r * self.exponential_base))
            self.s[arm] = logsumexp([self.s[arm], contrib_log])
        else:
            self.s[arm] += r

        # Update observation range for adaptive normalization
        self._update_obs_range(r)

    def posterior(self, subset=None):
        # Normalized mean rewards (exploitation)
        base = self._normalized_means(idx)

        # UCB bonus (exploration)
        t = total_pulls
        bonus = self.c * np.sqrt(2 * np.log(t) / pulls_per_arm)

        # Combined UCB score
        scores = base + bonus

        # ε-greedy selection
        winners = argmax(scores)
        probs[winners] = (1 - epsilon) / len(winners)
        probs[others] = epsilon / len(others)

        return probs
```

**Why This Matters**: Different LLMs excel at different stages/problems. Claude might be best for creative exploration, GPT-4 for rigorous verification. Dynamic selection automatically finds the right tool for each job.

---

### 4. Meta-Learning: Online Scratchpad

**Paper Concept**: Periodically summarizes successful solutions, extracting optimization strategies and design principles to guide future mutations.

#### How It Works:

**Scratchpad Updates** (`core/summarizer.py`):
- Every N generations (e.g., N=10), analyze recent successful programs
- LLM extracts patterns:
  - "What made these solutions work?"
  - "What common mistakes were avoided?"
  - "What design principles emerged?"
- Generates 3-5 bullet-point recommendations

**Prompt Injection** (`core/sampler.py`):
- When generating new mutations, LLM receives:
  - Parent code
  - Inspiration code (from archive)
  - **Meta-recommendations** from scratchpad
  - Task description
- Example: "Recent successful agents used ensemble methods with temperature=0.7 for diversity"

**Iterative Refinement**:
- Meta-recommendations updated continuously
- Old recommendations replaced if no longer relevant
- Best program always included in analysis

#### Code Implementation:

```python
# shinka/core/summarizer.py
class MetaSummarizer:
    def update_meta_memory(self, best_program):
        # Collect recent successful programs
        programs_to_analyze = self.evaluated_since_last_meta

        # Build prompt with program codes and metrics
        analysis_prompt = self._build_analysis_prompt(
            programs_to_analyze, best_program
        )

        # LLM extracts patterns
        response = self.meta_llm_client.query(
            msg=analysis_prompt,
            system_msg=META_SYSTEM_MSG
        )

        # Parse recommendations
        new_recommendations = self._parse_recommendations(response.content)

        # Store for future mutations
        self.current_recommendations = new_recommendations
        self.evaluated_since_last_meta = []

        return new_recommendations
```

**Why This Matters**: Evolution isn't just random - it learns what works. Meta-scratchpad enables the system to discover and reinforce successful patterns across generations.

---

## Code Implementation

### Architecture Overview

```
shinka/
├── core/
│   ├── runner.py          # Main evolution loop
│   ├── sampler.py         # LLM prompt generation
│   ├── summarizer.py      # Meta-learning scratchpad
│   └── novelty_judge.py   # Rejection sampling
├── database/
│   ├── dbase.py          # SQLite storage + archive
│   ├── parents.py        # Parent selection strategies
│   ├── inspirations.py   # Context selection
│   └── islands.py        # Island-based evolution
├── llm/
│   ├── llm.py            # Multi-LLM client
│   ├── dynamic_sampling.py # Bandit algorithms
│   └── embedding.py      # Code embeddings
└── edit/
    ├── apply_diff.py     # SEARCH/REPLACE patches
    └── apply_full.py     # Full code rewrites
```

### Main Evolution Loop

**File**: `shinka/core/runner.py:294-367`

```python
class EvolutionRunner:
    def run(self):
        # Generation 0: Initialize with seed program
        self._run_generation_0()

        # Main loop: Parallel job queue
        while self.completed_generations < target_generations:
            # 1. Check for completed jobs
            completed_jobs = self._check_completed_jobs()

            # 2. Process results
            for job in completed_jobs:
                self._process_completed_job(job)

                # Add to database and archive
                self.db.add(program)

                # Update meta-scratchpad if needed
                if self.meta_summarizer.should_update_meta(interval):
                    self.meta_summarizer.update_meta_memory()

            # 3. Submit new jobs
            if len(running_jobs) < max_parallel_jobs:
                self._submit_new_job()
```

**Key Function**: `_submit_new_job()` (`runner.py:616-760`)

```python
def _submit_new_job(self):
    # Novelty rejection sampling loop
    for novelty_attempt in range(max_novelty_attempts):
        # Resample loop (try different parents)
        for resample_attempt in range(max_patch_resamples):
            # 1. Sample parent + inspirations
            parent, archive_progs, topk_progs = self.db.sample(
                target_generation=current_gen,
                novelty_attempt=novelty_attempt
            )

            # 2. Generate code patch with LLM
            code_diff, metadata = self.run_patch(
                parent, archive_progs, topk_progs
            )

            if patch_successful:
                break

        # 3. Check code novelty
        code_embedding = self.get_code_embedding(exec_fname)

        should_accept = self.novelty_judge.assess_novelty(
            code_embedding, parent, self.db
        )

        if should_accept:
            break  # Success!
        # Otherwise: retry with different parent

    # 4. Submit evaluation job
    job_id = self.scheduler.submit_async(exec_fname, results_dir)
```

---

### Database and Archive System

**File**: `shinka/database/dbase.py`

**Core Data Structure**:

```python
@dataclass
class Program:
    id: str
    code: str
    parent_id: Optional[str]
    generation: int

    # Performance metrics
    correct: bool
    combined_score: float
    public_metrics: dict
    private_metrics: dict

    # Evolutionary metadata
    archive_inspiration_ids: List[str]
    top_k_inspiration_ids: List[str]
    embedding: List[float]

    # Island information
    island_idx: Optional[int]
    children_count: int = 0
```

**Archive (MAP-Elites Style)**:

```python
# dbase.py: _update_archive()
def _update_archive(self, program):
    # Archive stores best N programs per island
    island_archive = get_programs_in_island_archive(program.island_idx)

    if len(island_archive) < archive_size:
        # Archive not full - add program
        add_to_archive(program)
    else:
        # Replace worst program if new one is better
        worst = min(island_archive, key=lambda p: p.combined_score)
        if program.combined_score > worst.combined_score:
            remove_from_archive(worst)
            add_to_archive(program)
```

**Sampling Function** (`dbase.py:1475-1570`):

```python
def sample(self, target_generation, novelty_attempt=1):
    # 1. Select island for this generation
    island_idx = self.island_manager.get_island_for_generation(
        target_generation
    )

    # 2. Sample parent using configured strategy
    parent = self.parent_selector.sample_parent(island_idx)

    # 3. Sample archive inspirations (elite programs)
    archive_inspirations = self.context_selector.sample_archive_context(
        island_idx, num_inspirations=4
    )

    # 4. Sample top-k inspirations (recent good programs)
    topk_inspirations = self.context_selector.sample_topk_context(
        island_idx, k=2
    )

    return parent, archive_inspirations, topk_inspirations
```

---

### LLM Integration and Mutation

**Mutation Types** (`core/sampler.py`):

1. **Diff Patches (60%)** - SEARCH/REPLACE blocks:
```python
<SEARCH>
def forward(self, problem):
    response, cost = self.query_llm(problem)
    return response, cost
</SEARCH>

<REPLACE>
def forward(self, problem):
    # Try ensemble of 3 solutions
    solutions = []
    for i in range(3):
        resp, cost = self.query_llm(problem, temp=0.7)
        solutions.append(resp)
    # Majority vote
    final = majority_vote(solutions)
    return final, cost * 3
</REPLACE>
```

2. **Full Rewrites (30%)** - Complete code replacement:
```python
# EVOLVE-BLOCK-START
class Agent:
    def __init__(self, query_llm):
        self.query_llm = query_llm
        # Entirely new implementation

    def forward(self, problem):
        # Completely different approach
        pass
# EVOLVE-BLOCK-END
```

3. **Crossover (10%)** - Combine multiple parents:
```python
# Take ensemble strategy from Parent A
# Combine with verification from Parent B
# Add temperature sampling from Inspiration C
```

**Prompt Construction** (`core/sampler.py:sample()`):

```python
def sample(self, parent, archive_inspirations, topk_inspirations, meta_recs):
    # Build system message
    system_msg = f"{task_description}\n\nMETA-RECOMMENDATIONS:\n{meta_recs}"

    # Build user message
    user_msg = f"""
    PARENT CODE:
    {parent.code}

    PARENT PERFORMANCE: {parent.combined_score}

    ARCHIVE INSPIRATIONS (Elite Programs):
    {format_inspirations(archive_inspirations)}

    RECENT TOP PERFORMERS:
    {format_inspirations(topk_inspirations)}

    TASK: Improve the parent code. Generate a {patch_type} patch.
    """

    return system_msg, user_msg, patch_type
```

---

### Evaluation Wrapper

**File**: `shinka/core/wrap_eval.py:run_shinka_eval()`

```python
def run_shinka_eval(
    program_path,
    experiment_fn_name="run_experiment",
    num_runs=3,
    get_experiment_kwargs=None,
    aggregate_metrics_fn=None
):
    # 1. Load the program module dynamically
    spec = importlib.util.spec_from_file_location("module", program_path)
    module = importlib.util.module_from_spec(spec)

    # 2. Get the experiment function
    experiment_fn = getattr(module, experiment_fn_name)

    # 3. Run multiple trials
    results = []
    for run_idx in range(num_runs):
        kwargs = get_experiment_kwargs(run_idx)
        result = experiment_fn(**kwargs)
        results.append(result)

    # 4. Aggregate metrics across runs
    metrics = aggregate_metrics_fn(results)

    # 5. Compute combined score (fitness)
    combined_score = metrics["combined_score"]

    return metrics, correct=True, error=None
```

---

## AIME Case Study: End-to-End Example

### Problem Statement

**Task**: Design agent scaffolds for solving AIME (American Invitational Mathematics Examination) problems.

**Constraints**:
- Maximum 10 LLM queries per problem
- Base model: GPT-4.1-nano (weak reasoning model)
- Evaluation: 30 AIME 2024 problems
- Goal: Maximize accuracy (0-100%)

**Why This Is Hard**:
- AIME problems require multi-step reasoning
- Simple prompting achieves ~10-20% accuracy
- Need sophisticated scaffolding (ensembles, verification, etc.)
- Must work within tight query budget

---

### Configuration

**File**: `examples/adas_aime/run_evo.py`

```python
# Evolution parameters
evo_config = EvolutionConfig(
    num_generations=75,
    max_parallel_jobs=1,

    # Patch types: 60% diff, 30% full, 10% crossover
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],

    # Multiple LLMs for mutation generation
    llm_models=[
        "gemini-2.5-pro",              # Creative exploration
        "claude-sonnet-4",              # Rigorous reasoning
        "azure-o4-mini"                 # Fast iteration
    ],

    # Meta-learning: Update every 10 generations
    meta_rec_interval=10,
    meta_llm_models=["azure-gpt-4.1"],

    # Novelty rejection: 3 attempts, 0.95 threshold
    max_novelty_attempts=3,
    code_embed_sim_threshold=0.95,

    # Text feedback: Send failed problems to LLM
    use_text_feedback=True
)

# Database: 4 islands, archive size 40
db_config = DatabaseConfig(
    num_islands=4,
    archive_size=40,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,

    # Parent selection: Weighted strategy
    parent_selection_strategy="weighted",
    parent_selection_lambda=10.0,  # High λ = strong sigmoid

    # Migration every 10 generations
    migration_interval=10,
    migration_rate=0.1
)
```

---

### Initial Program (Generation 0)

**File**: `examples/adas_aime/initial.py:10-34`

```python
# EVOLVE-BLOCK-START
class Agent:
    def __init__(self, query_llm, temperature=0.0):
        self.output_format_instructions = (
            "Output only digits (0-999). "
            "Enclose answer in \\boxed{...}"
        )
        self.query_llm = query_llm
        self.temperature = temperature

    def forward(self, problem: str) -> tuple[str, float]:
        """Simple single-query baseline."""
        system_prompt = "You are a skilled mathematician."
        task_prompt = f"{self.output_format_instructions}\n\n{problem}"

        response, cost = self.query_llm(
            prompt=task_prompt,
            system=system_prompt,
            temperature=self.temperature
        )
        return response, cost
# EVOLVE-BLOCK-END
```

**Generation 0 Performance**: ~15-20% accuracy (baseline)

---

### Evolution Process

**Generation 1-10: Early Exploration**

Discovered patterns:
- Chain-of-thought prompting (Gen 3)
- Multiple temperature sampling (Gen 5)
- Simple majority voting (Gen 7)

**Meta-Recommendation (Gen 10)**:
```
1. Temperature diversity (0.0, 0.5, 1.0) significantly improves ensemble quality
2. Majority voting with 3+ solutions outperforms single solutions
3. Explicit "think step-by-step" instructions increase accuracy
4. Format errors reduced by clear output instructions
```

**Generation 11-30: Refinement**

Key improvements:
- Expert personas introduced (Gen 15)
- Self-critique mechanisms (Gen 18)
- Fallback logic for failed queries (Gen 22)

**Generation 31-50: Advanced Architectures**

Breakthrough (Gen 35):
- **Three-stage pipeline**: Generation → Review → Synthesis
- Uses 7/10 query budget efficiently
- Achieves 32% accuracy

**Generation 51-75: Optimization**

Final refinements:
- Peer review with pattern verification (Gen 58)
- Independent expert personas (Gen 62)
- Robust fallback mechanisms (Gen 70)

---

### Discovered Solution (Generation 10)

**File**: `examples/adas_aime/discovered/2_gen10_expert_ensemble_with_self_correction.py`

**Architecture**: Three-stage reasoning with diverse expert personas

```python
class Agent:
    def __init__(self, query_llm, temperature=0.0):
        self.generation_temperature = 0.7
        self.review_temperature = 0.1
        self.synthesis_temperature = 0.0

        # Three distinct expert personas
        self.expert_personas = [
            "Meticulous mathematician: slow, careful, step-by-step",
            "Intuitive mathematician: elegant, pattern-seeking",
            "Algorithmic mathematician: computational, state-based"
        ]

    def forward(self, problem: str) -> tuple[str, float]:
        total_cost = 0.0

        # === STAGE 1: Diverse Generation (3 queries) ===
        solutions = []
        for persona in self.expert_personas:
            response, cost = self.query_llm(
                prompt=f"Solve: {problem}",
                system=persona,
                temperature=0.7  # High temp for diversity
            )
            solutions.append(response)
            total_cost += cost

        # === STAGE 2: Independent Peer Review (3 queries) ===
        reviewer_prompt = (
            "You are a skeptical peer reviewer. "
            "Check calculations. Test patterns on new examples. "
            "Identify errors and provide corrections."
        )

        critiques = []
        for solution in solutions:
            review, cost = self.query_llm(
                prompt=f"Problem: {problem}\nSolution: {solution}\nReview:",
                system=reviewer_prompt,
                temperature=0.1  # Low temp for careful review
            )
            critiques.append(review)
            total_cost += cost

        # === STAGE 3: Synthesis (1 query) ===
        synthesis_prompt = f"""
        You are the Editor-in-Chief. Produce the definitive solution.

        Problem: {problem}

        ATTEMPT 1:
        Solution: {solutions[0]}
        Critique: {critiques[0]}

        ATTEMPT 2:
        Solution: {solutions[1]}
        Critique: {critiques[1]}

        ATTEMPT 3:
        Solution: {solutions[2]}
        Critique: {critiques[2]}

        Synthesize the final, correct answer.
        """

        final_response, cost = self.query_llm(
            prompt=synthesis_prompt,
            system="Master mathematician and editor",
            temperature=0.0  # Deterministic synthesis
        )
        total_cost += cost

        # === Fallback: Majority Vote ===
        if self._extract_answer(final_response) is None:
            # Use reviewed answers
            reviewed_answers = [
                self._extract_answer(c) for c in critiques
            ]
            valid_answers = [a for a in reviewed_answers if a]

            if valid_answers:
                most_common = Counter(valid_answers).most_common(1)[0][0]
                final_response += f"\n\\boxed{{{most_common}}}"

        return final_response, total_cost
```

**Query Budget**: 3 + 3 + 1 = **7 queries** (within 10 limit)

**Performance**: **34.4% accuracy** (>70% improvement over baseline)

**Key Innovations**:
1. **Diverse Expert Personas**: Reduces correlated errors
2. **Independent Peer Review**: Each solution gets critical analysis
3. **Pattern Verification**: Reviewers explicitly test patterns on new examples
4. **Informed Synthesis**: Final agent sees all solutions + critiques
5. **Robust Fallbacks**: Majority voting if synthesis fails

---

### Example Evolution Timeline

| Gen | Event | Accuracy | Key Change |
|-----|-------|----------|------------|
| 0 | Initial program | 15% | Single query baseline |
| 3 | First mutation | 18% | Added "think step-by-step" |
| 7 | Ensemble discovered | 25% | 3-solution majority vote |
| 10 | Meta-update #1 | - | "Temperature diversity works" |
| 15 | Expert personas | 28% | Different system prompts |
| 20 | Meta-update #2 | - | "Review mechanisms help" |
| 25 | Review stage added | 31% | Critique before synthesis |
| 30 | Meta-update #3 | - | "7-query pipeline optimal" |
| 35 | **Breakthrough** | 34% | Full 3-stage architecture |
| 50 | Refinement | 34.2% | Better fallback logic |
| 75 | Final | 34.4% | Robust to edge cases |

---

### Evaluation Process

**File**: `examples/adas_aime/evaluate.py`

```python
def main(program_path, model_name="gpt-4.1-nano", year=2024):
    # Run 3 independent trials
    results = run_shinka_eval(
        program_path=program_path,
        experiment_fn_name="run_experiment",
        num_runs=3,
        get_experiment_kwargs=lambda i: {
            "model_name": model_name,
            "year": year,
            "max_calls": 10  # Enforce query limit
        }
    )

    # Aggregate metrics
    metrics = {
        "combined_score": mean_accuracy,  # Fitness for evolution
        "public": {
            "cost": mean_cost,
            "avg_num_llm_calls": mean_calls
        },
        "text_feedback": construct_text_feedback(failed_problems)
    }

    return metrics
```

**Text Feedback Example** (sent to LLM in next generation):

```
# Example of AIME problem that could not be answered correctly:

In a tournament, 16 players play each other exactly once...

# Agent's wrong response:
Let's use combinatorics. Total games = C(16,2) = 120...
[calculates based on incorrect assumption]

# Agent's submit answer: 342

# Ground truth: 455

[This feedback helps LLM understand what went wrong]
```

---

### Why This Works: Connecting Paper to Practice

**1. Adaptive Parent Sampling** (Weighted Strategy):
- Early generations: Explores diverse approaches (single query, CoT, ensemble)
- Later generations: Exploits successful patterns (3-stage architecture)
- Novelty bonus: Prevents over-exploiting the 7-query solution, discovers 10-query variants too

**2. Novelty Rejection Sampling**:
- Prevents generating 20 variants of "3-solution majority vote"
- Forces exploration of different architectures (review, synthesis, verification)
- Saves ~30-40% of evaluations

**3. Bandit-Based LLM Selection**:
- Gemini-2.5-pro: Best for creative generation stage (high diversity)
- Claude Sonnet-4: Best for critical review stage (rigorous analysis)
- GPT-4o-mini: Fast iteration for minor refinements
- System automatically learns this assignment

**4. Meta-Learning**:
- Gen 10: "Temperature diversity improves ensembles"
- Gen 20: "Review mechanisms catch errors"
- Gen 30: "7-query pipeline is optimal trade-off"
- These insights guide later mutations toward successful patterns

---

### Results and Generalization

**On AIME 2024 (Training Set)**:
- Initial: 15% accuracy
- Final: 34.4% accuracy
- Improvement: +129% relative

**Generalization Test (AIME 2025)**:
- Performance: 36% accuracy
- **Better than training set** (suggests genuine learning, not overfitting)

**Cross-LLM Transfer**:
- Evolved scaffold designed for GPT-4.1-nano
- Also tested on: GPT-4.1-mini, GPT-4.1, o4-mini
- **Performance improved across all models** (architecture transfers!)

**Computational Efficiency**:
- Discovered in 75 generations
- ~150 total evaluations (75 gen × 2 parallel jobs)
- Cost: ~$50 in API calls
- Compare to: Thousands of evaluations in prior work

---

### Key Takeaways

1. **Sample Efficiency**: ShinkaEvolve finds state-of-the-art solutions with ~100x fewer evaluations than baseline evolutionary methods

2. **Open-Ended Discovery**: Doesn't just optimize parameters - discovers novel architectural patterns (multi-stage pipelines, expert personas, peer review)

3. **Adaptive and Robust**: Automatically balances exploration/exploitation, prevents redundancy, selects best LLMs for each task

4. **Practical and Accessible**: Open-source, works with any LLM API, requires minimal compute (runs on laptop)

5. **Broadly Applicable**: Same framework used for:
   - Agent scaffolding (AIME)
   - Algorithm optimization (circle packing)
   - Code optimization (ALE-Bench)
   - Neural network training (MoE load balancing)

---

## Conclusion

ShinkaEvolve demonstrates that **LLM-driven evolution can efficiently discover sophisticated solutions** to complex optimization problems. By combining:

- **Smart parent sampling** (exploit good solutions, explore new ideas)
- **Novelty rejection** (avoid wasting resources on duplicates)
- **Adaptive LLM selection** (use the right model for each task)
- **Meta-learning** (accumulate and apply insights across generations)

The system achieves **state-of-the-art results with minimal computational resources**.

The AIME case study illustrates the complete pipeline: from a simple baseline (15% accuracy) to a sophisticated three-stage reasoning architecture (34% accuracy) through automated evolution - no human intervention required after setup.

This represents a **paradigm shift in program optimization**: instead of manually designing algorithms, we can specify the goal and let evolution discover the solution.

---

## References

**Paper**: [ShinkaEvolve: Open-Ended and Sample-Efficient Program Evolution](https://arxiv.org/html/2509.19349v1)

**Code Repository**: [github.com/saptarshiguha/ShinkaEvolve](https://github.com/saptarshiguha/ShinkaEvolve)

**Key Implementation Files**:
- Evolution loop: `shinka/core/runner.py`
- Parent sampling: `shinka/database/parents.py`
- Novelty rejection: `shinka/core/novelty_judge.py`
- LLM bandit: `shinka/llm/dynamic_sampling.py`
- Meta-learning: `shinka/core/summarizer.py`
- AIME example: `examples/adas_aime/`
