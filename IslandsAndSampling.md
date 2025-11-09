# Understanding Islands and Sampling in ShinkaEvolve

A simple, intuitive explanation of how ShinkaEvolve's population management and sampling works.

---

## The Island Concept: Why Multiple Populations?

### The Problem: Premature Convergence

Imagine you're searching for the best solution to a problem, and you find a pretty good one early on. Natural tendency is to keep refining that solution. But what if there's a completely different approach that's even better, but you'll never discover it because you're too focused on improving what you already have?

This is called **premature convergence** - getting stuck in a "local optimum" (a good solution) and missing the "global optimum" (the best solution).

### The Solution: Island-Based Evolution

**Paper Reference**: Section 3.1 mentions that ShinkaEvolve uses "a population-based approach with islands" to maintain diversity.

Think of islands like **parallel laboratories**, each exploring different approaches:

```
Island 0: Focusing on ensemble methods
Island 1: Exploring chain-of-thought prompting
Island 2: Trying verification strategies
Island 3: Experimenting with temperature sampling
```

**How It Works** (`database/islands.py:66-110`):

1. **Generation 0 (Initialization)**:
   - The initial program is created or loaded
   - It's placed on Island 0
   - **Copies are made for all other islands** (Islands 1, 2, 3...)
   - Each island now has the same starting point

2. **Independent Evolution**:
   - Each island evolves its programs **separately**
   - Island 0 might discover ensemble methods
   - Island 1 might discover self-verification
   - They don't interfere with each other initially

3. **Why This Matters**:
   - Different islands can explore radically different approaches
   - If one island gets stuck, others keep exploring
   - Diversity is preserved across the entire population

**Real Example from AIME**:
- Island 0 evolved programs using 3 LLM queries with majority voting
- Island 2 evolved programs using 7 LLM queries with expert personas
- Both approaches were valuable, and neither dominated prematurely

---

## How Sampling Works: Choosing What to Improve

When ShinkaEvolve needs to create a new program, it goes through a **three-step sampling process**:

### Step 1: Choose a Parent Program

**Paper Reference**: Section 3.1 discusses "adaptive parent sampling" as one of the three key innovations.

The **parent** is the program that will be modified to create the new program. Think of it as the "base" you're starting from.

**How Parents Are Selected** (`database/parents.py:274-455`):

**Weighted Selection Strategy**:
- Each program in the archive gets a **weight** that determines its selection probability
- The weight combines two factors:
  1. **Performance**: How well did it solve the problem?
  2. **Novelty**: Has it been used as a parent before, or is it fresh?

**In Simple Terms**:
```
High-performing program + Few children = High weight (likely to be selected)
High-performing program + Many children = Lower weight (give others a chance)
Low-performing program = Low weight (unlikely to be selected)
```

**Why Both Factors Matter**:
- **Performance alone** would mean using the same best program over and over (exploitation)
- **Novelty alone** would mean trying random programs (pure exploration)
- **Combining them** balances finding better solutions while exploring new directions

**Code Example** (`database/parents.py:404-420`):
```python
# For each program in the archive:
performance_score = sigmoid(program.score - median_score)  # How good is it?
novelty_bonus = 1 / (1 + program.children_count)          # How fresh is it?
weight = performance_score × novelty_bonus                # Combined weight

# Then sample using these weights as probabilities
```

**Island Constraint**: Parents are selected **only from the same island**. If we're generating a program for Island 2, we only look at Island 2's programs.

---

### Step 2: Choose Inspiration Programs

**Paper Reference**: Section 3.1 mentions using "context programs" from the archive to inspire mutations.

After selecting a parent, ShinkaEvolve also provides the LLM with **inspiration programs** - examples of other successful solutions to learn from.

**Two Types of Inspirations** (`database/inspirations.py`):

**2a. Archive Inspirations (Elite Programs)**:
- The **archive** is a collection of the best programs discovered so far
- Think of it like a "hall of fame" - only the top performers get in
- ShinkaEvolve samples **4 programs** from the archive (by default)
- These are from the **same island** as the parent

**How the Archive Works** (`database/dbase.py:_update_archive`):
```
When a new correct program is evaluated:
  If archive is not full:
    Add program to archive
  Else if new program is better than worst program in archive:
    Remove worst program
    Add new program
```

**Archive Size**: Configurable (default 40 programs per island in AIME example)

**2b. Top-K Inspirations (Recent Good Programs)**:
- The **2 most recent** high-performing programs (by default)
- These are newer discoveries that might not be in the archive yet
- Provides "fresh ideas" alongside the classic "hall of fame" examples

**Why Both Types?**:
- **Archive**: Battle-tested, proven approaches
- **Top-K**: Cutting-edge, recent breakthroughs
- Together: Stable foundation + innovative ideas

---

### Step 3: Put It All Together

**What the LLM Receives** (`core/sampler.py:sample()`):

```
PARENT PROGRAM:
[Code of the selected parent]
Performance: 28% accuracy

ELITE INSPIRATIONS FROM ARCHIVE:
Inspiration 1: 32% accuracy - Uses ensemble of 3 solutions
Inspiration 2: 30% accuracy - Uses verification step
Inspiration 3: 29% accuracy - Uses expert personas
Inspiration 4: 27% accuracy - Uses temperature sampling

RECENT TOP PERFORMERS:
Recent 1: 31% accuracy - Combines verification + ensemble
Recent 2: 30% accuracy - Uses fallback logic

META-RECOMMENDATIONS (if available):
1. Temperature diversity (0.0, 0.5, 1.0) improves results
2. Peer review catches errors
3. 7-query budgets work well

YOUR TASK: Improve the parent program using ideas from the inspirations
```

**The LLM's Job**:
- Look at what made the parent successful
- Learn from the inspiration programs
- Generate a modification that combines good ideas in a new way

---

## Choosing Modification Types: Small Tweaks vs Big Rewrites

**Paper Reference**: Section 3.1 mentions three mutation operators: "SEARCH/REPLACE edits, full rewrites, and crossover."

ShinkaEvolve doesn't just do one type of modification - it randomly chooses from three different strategies.

### The Three Modification Types

**Configuration** (`examples/adas_aime/run_evo.py:55-56`):
```python
patch_types = ["diff", "full", "cross"]
patch_type_probs = [0.6, 0.3, 0.1]
```

This means:
- 60% chance: Small targeted edit (diff)
- 30% chance: Complete rewrite (full)
- 10% chance: Combine multiple programs (cross)

---

### Type 1: Diff Patches (60% probability)

**What It Does**: Makes **small, surgical changes** to specific parts of the code.

**Paper Reference**: Section 3.1 mentions "SEARCH/REPLACE" edits as one mutation type.

**Format** (`edit/apply_diff.py`):
```python
<SEARCH>
def forward(self, problem):
    response = self.query_llm(problem)
    return response
</SEARCH>

<REPLACE>
def forward(self, problem):
    # Try 3 solutions with temperature sampling
    solutions = []
    for temp in [0.0, 0.5, 1.0]:
        response = self.query_llm(problem, temperature=temp)
        solutions.append(response)
    return self.majority_vote(solutions)
</REPLACE>
```

**When This Works Well**:
- The parent is already pretty good
- You want to add one specific feature
- You're refining an existing approach
- Example: Adding a verification step to an existing ensemble

**Code Location**: `edit/apply_diff.py:apply_diff_patch()`

---

### Type 2: Full Rewrites (30% probability)

**What It Does**: **Completely replaces** the code within the evolvable section.

**Format** (`edit/apply_full.py`):
```python
# EVOLVE-BLOCK-START
class Agent:
    def __init__(self, query_llm):
        # COMPLETELY NEW IMPLEMENTATION
        self.expert_personas = [...]  # New approach!

    def forward(self, problem):
        # COMPLETELY DIFFERENT STRATEGY
        # Three-stage pipeline instead of simple query
        stage1 = self.generate_solutions(problem)
        stage2 = self.peer_review(stage1)
        stage3 = self.synthesize(stage2)
        return stage3
# EVOLVE-BLOCK-END
```

**When This Works Well**:
- Need to explore radically different approaches
- Current parent has reached its limits
- Want to try architectural changes
- Example: Switching from single-query to multi-stage pipeline

**Code Location**: `edit/apply_full.py:apply_full_patch()`

---

### Type 3: Crossover (10% probability)

**What It Does**: **Combines features** from multiple programs (parent + inspirations).

**Paper Reference**: Section 3.1 mentions "crossover" as a mutation operator.

**Conceptual Example**:
```
Parent: Uses ensemble of 3 solutions
Inspiration 1: Uses verification step
Inspiration 2: Uses expert personas

Crossover Result:
→ Ensemble of 3 expert personas, each verified independently
```

**When This Works Well**:
- Multiple good ideas exist separately
- Combining them might create synergy
- Example: Merge "temperature sampling" + "self-verification"

**Code Location**: `core/sampler.py` (prompts LLM to combine programs)

---

## Island Migration: Sharing Good Ideas

**Paper Reference**: Section 3.1 mentions island-based evolution with periodic migration.

Even though islands evolve independently, they occasionally **share their best discoveries**.

### How Migration Works (`database/islands.py:213-288`)

**Timing**: Every N generations (default: every 10 generations in AIME)

**Process**:

1. **Select Migrants**:
   - From each island, pick ~10% of programs (configurable)
   - **Protected**: Generation 0 programs never migrate (they're the foundation)
   - **Protected**: The best program on each island never migrates (elitism)
   - Only **correct programs** migrate (no point spreading failures)

2. **Choose Destination**:
   - Each selected program is sent to a **random different island**
   - Program is **moved** (not copied) to the new island
   - Migration history is tracked

3. **Why Migration Matters**:
   - Island 0 discovers a great ensemble method → migrates to Island 2
   - Island 2 can now build on that ensemble idea
   - Good ideas spread while islands remain mostly independent

**Real Example**:
```
Generation 20 Migration:
Island 0 → Island 3: Program with score 0.32 (ensemble method)
Island 1 → Island 2: Program with score 0.30 (verification)
Island 2 → Island 0: Program with score 0.28 (expert personas)
Island 3 → Island 1: Program with score 0.27 (fallback logic)
```

Now each island has access to ideas from other islands while maintaining its own direction.

---

## Complete Sampling Flow: A Concrete Example

Let's walk through **Generation 15** on **Island 2** in the AIME experiment:

### Before Sampling Starts

**Island 2's Archive** (40 programs):
- Best program: 34% accuracy (7-query expert ensemble)
- 2nd best: 32% accuracy (5-query verification)
- 3rd best: 31% accuracy (3-query majority vote)
- ... (37 more programs)

### Step 1: Sample Parent

**Weighted Selection** (`database/parents.py:274-455`):
```
Calculate weights for all 40 archive programs:

Program 1 (34% accuracy, 8 children):
  performance_score = sigmoid((34 - 28) / 3.2) = 0.85  # High performance
  novelty_bonus = 1 / (1 + 8) = 0.11                   # Many children (used often)
  weight = 0.85 × 0.11 = 0.094

Program 2 (32% accuracy, 2 children):
  performance_score = sigmoid((32 - 28) / 3.2) = 0.77  # Good performance
  novelty_bonus = 1 / (1 + 2) = 0.33                   # Few children (fresh!)
  weight = 0.77 × 0.33 = 0.254                         # HIGHEST WEIGHT!

Program 3 (31% accuracy, 5 children):
  performance_score = sigmoid((31 - 28) / 3.2) = 0.72
  novelty_bonus = 1 / (1 + 5) = 0.17
  weight = 0.72 × 0.17 = 0.122

... (37 more)
```

**Result**: Program 2 is selected (highest weight due to good performance + freshness)

### Step 2: Sample Inspirations

**From Archive** (`database/inspirations.py`):
- Randomly select 4 programs from Island 2's archive
- Selected: Programs with scores 34%, 31%, 30%, 29%

**From Top-K**:
- Get the 2 most recent high-performing programs
- Selected: Programs from generations 14 and 13 with scores 32%, 31%

### Step 3: Choose Modification Type

**Random Sample** (`core/sampler.py`):
```python
patch_type = random.choices(
    ["diff", "full", "cross"],
    weights=[0.6, 0.3, 0.1]
)[0]
```

**Result**: "diff" is selected (60% chance)

### Step 4: Build Prompt for LLM

**The LLM Receives** (`core/sampler.py:sample()`):

```
SYSTEM MESSAGE:
You are an expert ML engineer improving agent scaffolds for AIME problems.

META-RECOMMENDATIONS:
1. Expert personas with temperature 0.7 work well
2. Peer review mechanisms catch errors
3. 7-query budgets optimal

USER MESSAGE:
PARENT PROGRAM (32% accuracy, Generation 12):
[Shows code with 5-query verification strategy]

ARCHIVE INSPIRATIONS:
Inspiration 1 (34%): [7-query expert ensemble]
Inspiration 2 (31%): [3-query majority vote]
Inspiration 3 (30%): [Single query with CoT]
Inspiration 4 (29%): [Temperature sampling]

TOP-K INSPIRATIONS:
Recent 1 (32%): [Verification + ensemble]
Recent 2 (31%): [Expert personas + fallback]

Generate a DIFF patch to improve the parent program.
Use SEARCH/REPLACE blocks to make targeted improvements.
```

### Step 5: LLM Generates Modification

**LLM Response**:
```
<NAME>verification_with_expert_personas</NAME>
<DESCRIPTION>
Enhance the parent's verification strategy by adding expert personas
from the archive inspirations, combining the 32% verification approach
with the 34% expert ensemble concept.
</DESCRIPTION>

<SEARCH>
response = self.query_llm(problem)
</SEARCH>

<REPLACE>
# Use expert persona instead of generic prompt
expert_prompt = "You are a methodical mathematician..."
response = self.query_llm(problem, system=expert_prompt, temp=0.7)
</REPLACE>
```

### Step 6: Apply Modification

**Patch Application** (`edit/apply_diff.py`):
1. Find the SEARCH block in parent code
2. Replace with REPLACE block
3. Verify syntax is valid
4. Save new program to `gen_15/main.py`

### Step 7: Evaluate New Program

**Evaluation** (`core/runner.py:_submit_new_job`):
1. Run evaluation script (3 independent trials)
2. Calculate metrics: 33% accuracy
3. Store in database with parent=Program2, generation=15, island_idx=2

### Step 8: Update Archive

**Archive Update** (`database/dbase.py:_update_archive`):
```
New program: 33% accuracy
Worst in archive: 26% accuracy

Since 33% > 26%:
  Remove worst program
  Add new program to archive
```

Now this new program is available for future sampling!

---

## Why This Approach Works

### 1. **Balance of Exploration and Exploitation**

**Exploitation** (Using what works):
- Weighted sampling favors high-performing parents
- Archive stores proven approaches
- Diff patches refine existing solutions

**Exploration** (Trying new things):
- Novelty bonus prevents overusing same parents
- Islands explore different directions independently
- Full rewrites enable radical changes
- Migration spreads diverse ideas

**Paper Reference**: Section 3.1 discusses this balance as key to sample efficiency.

### 2. **Multi-Scale Search**

**Small Changes** (Diff - 60%):
- Fine-tune existing solutions
- Quick iterations
- Low risk

**Medium Changes** (Full - 30%):
- Architectural experiments
- Higher risk, higher reward

**Recombination** (Cross - 10%):
- Combine proven ideas
- Synergistic discoveries

### 3. **Diversity Preservation**

**Islands**:
- 4 independent populations
- Different evolutionary trajectories
- Prevents entire population converging to one solution

**Migration**:
- Share breakthroughs across islands
- Not too frequent (every 10 gens) - lets islands develop unique approaches
- Elite protection - keeps best solutions safe

---

## Key Takeaways

1. **Islands are parallel laboratories** - each explores different approaches independently

2. **Sampling is three-tiered**:
   - **Parent**: One program to modify (weighted by performance + novelty)
   - **Archive inspirations**: 4 elite programs to learn from
   - **Top-K inspirations**: 2 recent successes to incorporate

3. **Modification types vary**:
   - 60% small edits (diff)
   - 30% complete rewrites (full)
   - 10% combinations (cross)

4. **Migration shares ideas** between islands every N generations (elite programs protected)

5. **Everything is island-scoped** - sampling, archives, and evolution happen within each island until migration

This design achieves **sample efficiency** (finds good solutions quickly) while maintaining **open-ended exploration** (doesn't get stuck in local optima).

---

## Code References

| Concept | File | Function/Class |
|---------|------|----------------|
| Island assignment | `database/islands.py` | `CopyInitialProgramIslandStrategy.assign_island()` (lines 128-190) |
| Island migration | `database/islands.py` | `ElitistMigrationStrategy.perform_migration()` (lines 216-288) |
| Parent sampling | `database/parents.py` | `WeightedSamplingStrategy.sample_parent()` (lines 277-455) |
| Archive inspirations | `database/inspirations.py` | `ArchiveContextSelector.sample_context()` |
| Top-K inspirations | `database/inspirations.py` | `TopKContextSelector.sample_context()` |
| Patch type selection | `core/sampler.py` | `PromptSampler.sample()` (random.choices) |
| Diff application | `edit/apply_diff.py` | `apply_diff_patch()` (lines 1-770) |
| Full rewrite | `edit/apply_full.py` | `apply_full_patch()` (lines 1-303) |
| Complete sampling flow | `database/dbase.py` | `ProgramDatabase.sample()` (lines 1475-1570) |
