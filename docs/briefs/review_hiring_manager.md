# Brief: hiring-manager review of the repository (1 October 2026)

You are reviewing a public GitHub repository as the hiring manager for a quantitative research / ML engineering team (think: head of research at a systematic trading firm or an anti-fraud ML lead), deciding whether to interview the owner and what to probe. Read-only; you may run the unit tests and import modules, but run no training or experiment scripts.

Read: README.md; docs/method.md; docs/experiments.md (the first 300 lines, the entries 109, 114, 131b, 140–145, 152–158 and the last 200 lines carefully, skim the rest); docs/research/ (skim); src/structural_break/white.py, combiners.py, and one more module of your choice; scripts/assemble_submission.py and scripts/ship_submission.sh; tests/; submissions/089-joint-nets/interface_089.py; CLAUDE.md.

Write the review in English, concise and specific, with file:line references:
1. Technical substance: is the statistics right (whitening, CUSUM/GLR, Shiryaev–Roberts, BOCPD usage, the per-step AUC metric and its consequences)? Any claims that are wrong, overstated, or unsupported by the journal's numbers? Any sign of leakage or of tuning on the evaluation fold beyond what the text admits?
2. Engineering quality: code structure, the batch-vs-streaming verification idea, the assembler/ship protocol, tests, reproducibility (can a reader rerun anything? what is missing), dead code or portfolio-unfriendly leftovers, the size and readability of the journal.
3. Research quality: is the experiment log a real scientific record (pre-stated kill conditions, controls, like-for-like comparisons) or a diary? Are the conclusions (the independent-member principle, the transfer rule, the ceiling anatomy) supported? What would you challenge in an interview?
4. The human/LLM question: what evidence in the record tells you what the human actually contributed and whether that is a skill you would hire for? What is missing to make that credible? (The text must stay truthful.)
5. Portfolio presentation: does the README land the value in the first screen? Is it too long? Which figures work and which do not? Which sections would you cut, move or add?
6. Verdict: interview or not, the level you would consider, the five interview questions you would ask based on this repository, and the top five concrete fixes before the owner makes it public.
