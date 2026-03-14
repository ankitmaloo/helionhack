# AlphaEvolve Paper Study (MVP-Oriented)

This note was generated from `alphaevolve/alphaevolve.pdf` for implementation grounding.

## MVP Extraction Summary
- Evolvable code is marked with `# EVOLVE-BLOCK-START` / `# EVOLVE-BLOCK-END`.
- Mutations are emitted as `SEARCH/REPLACE` diffs.
- Evaluators return scalar metrics and can use cascades.
- Evolution reuses prior high-quality ideas with exploration/exploitation balance.
- Main loop: sample parent + inspirations -> mutate -> evaluate -> add back to DB.

## Evidence Snippets
### EVOLVE Blocks and API
gnfacilitatesintegratingitwithexistingcodebaseswhilerequiring only minimal changes, simply by adding special markers (# EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END) as comments into the code. 4 AlphaEvolve: A coding agent for scientific and algorithmic discovery Any user-provided code inside such evolution blocks serves as the initial solution to be improved byAlphaEvolve, and the rest of the code forms a skeleton that ties the evolved pieces together, so that they can be invoked fromevaluate. While this initial implementation must be complete, it can be rudimentary—for instance, consisting of single-line functions that return constants of the appropriate types.

### Prompt Sampling
concise [83], whereas for problems with non-symmetric solutions it works better to evolve customized search algorithms. 2.2. Prompt sampling AsAlphaEvolveleveragesSOTALLMs, itsupportsvarioustypesofcustomizationandproviding long contexts as part of the primary evolution prompt. This prompt comprises multiple previously discovered solutions sampled from the program database, as well as system instructions on how to propose changes to a particular solution. Beyond these key ingredients, users can further tailor prompts to their specific needs in different ways, such as the following. • Explicit context: details about the problem being solved, such as fixed human-wr

### SEARCH/REPLACE Format
btain the scores of the newly proposed program. 6 AlphaEvolve: A coding agent for scientific and algorithmic discovery Output format. When AlphaEvolveasks an LLM to modify existing code, especially within larger codebases, it requests the changes to be provided as a sequence of diff blocks in a specific format: <<<<<<< SEARCH # Original code block to be found and replaced ======= # New code block to replace the original >>>>>>> REPLACE Here, the code between<<<<<<< SEARCH and ======= is the exact segment to match in the current program version. The code between======= and >>>>>>> REPLACE is the new segment that will replace the original one. This allows for tar

### Evaluation Cascade
ution. In practice,AlphaEvolve supports optional mechanisms to make this evaluation more flexible and more efficient: • Evaluation cascade (hypothesis testing): the user can specify ensembles of test cases of increasing difficulty, such that new solutions are evaluated on the next stage only if they achieve sufficiently promising results in all earlier stages. This helps to prune out less promising solutions more quickly. Moreover, new solutions are initially evaluated on a small scale before being subjected to the main test cases, to filter out faulty programs early. • LLM-generated feedback: in some applications, desirable solutions have certain charac- terist

### Multi-Metric Optimization
ized initializations), allowingAlphaEvolveto distribute this work through asynchronous calls to an evaluation cluster. Multiple scores. AlphaEvolve allows for optimizing multiple user-provided scores, i.e., evolving objects that achieve a high score under one or multiple evaluation metrics. This has both an intrinsic and instrumental value. While in multiple applications we genuinely care about developing solutions for multiple evaluation metrics (or one solution that is strong on all of them simultaneously), we find that even if one metric is of particular interest, optimizing for multiple metrics often improves results for the single target metric. Perhaps th

### Evolutionary Database
date solutions, increasing the chances of discovering novel approaches that are highly effective for the target metric. 2.5. Evolution During its evolutionary procedure,AlphaEvolvecontinually generates a growing number of solutions with evaluation results (scores and program outputs) attached to them. These solutions are stored in an evolutionary database, the primary goal of which is to optimally resurface previously explored ideas in future generations. A key challenge in designing such databases is balancing exploration and exploitation, to continuously improve the best programs while maintaining diversity to encourage exploration of the entire search space.

### Controller Loop
ce of existing or custom LLMs Scientist / Engineer Best program AlphaEvolve Evaluation code Distributed Controller Loop parent_program, inspirations = database.sample() prompt = prompt_sampler.build(parent_program, inspirations) diff = llm.generate(prompt) child_program = apply_diff(parent_program, diff) results = evaluator.execute(child_program) database.add(child_program, results) Evaluators pool LLMs ensemble Prompt sampler Program database Figure 2 | Expanded view of theAlphaEvolvediscovery process. The user provides an initial program (with components to evolve marked), evaluation code, and optional configurations (Section 2.1). AlphaEvolve then initiates
