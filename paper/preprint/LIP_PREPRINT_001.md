---
title: Receiver-Anchored Tests for Latent Communication
subtitle: Constant-Capacity Causal Localization and a Negative Cross-Model Result
author: Cristiano Silva
affiliation: Independent Researcher, Brazil
correspondence: ziwehdafe@gmail.com
project: LIP — Latent Injection Protocol
version: 0.1
date: 15 August 2026
doi: https://doi.org/10.5281/zenodo.21943476
repository: https://github.com/zigfreud/latent-llm-agent-communication
status: Preprint
license: Manuscript CC BY 4.0 · Code MIT
---

# Abstract

Latent communication between language models is often evaluated as a single end-to-end problem: a sender state is mapped into a receiver, and downstream accuracy is treated as evidence for or against the whole idea. This conflates at least two questions. First, can the selected receiver-side carrier express task identity at all? Second, can a learned cross-model bridge reconstruct the task-specific receiver state well enough to affect generation? We formalize and apply a receiver-anchored evaluation protocol that separates these questions with receiver-native oracle replay, same-capacity identity shuffles, sealed task-level confirmation, and functional execution controls.

In a preregistered same-model study on 32 capability-screened MBPP tasks, a 24-position residual packet replayed through the first eight blocks of a pinned Llama-3-8B-Instruct receiver achieved 79/96 functional passes (82.29%), while an equal-capacity same-stratum donor packet achieved 0/96. A constant-capacity 2 x 2 x 2 source factorial localized confirmed positive contributions to the terminal instruction core and function-name states (Holm-adjusted one-sided p = 0.0000699993 for each); no positive boundary contribution was detected under the registered contrast.

We then tested a preregistered heterogeneous DeepSeek-Coder-1.3B-to-Llama-3-8B learned residual-packet bridge. All three primary replicas passed held-out retrieval and margin gates, and the receiver-native oracle anchor again succeeded (84/96 versus 0/96 shuffled). Nevertheless, learned matched and learned shuffled packets both achieved 1/288 functional passes, yielding a task-level identity effect of zero. The result is negative for this complete registered system. It shows that geometric separability on held-out development tasks was insufficient evidence of causal functional transport. The broader contribution is an auditable evaluation pattern for distinguishing carrier failure, bridge failure, and unsupported universality claims.

Keywords: latent communication; language-model agents; activation intervention; residual stream; cross-model alignment; causal evaluation; negative results; reproducible AI research

# 1. Introduction

Language-model agents normally communicate through text. Direct communication through activations, hidden states, or key-value caches could in principle avoid repeated decoding and preserve information that text serialization discards. The central technical challenge is not merely producing a continuous vector that resembles a target representation. It is demonstrating that task-specific information survives the complete sender-to-receiver path and changes downstream behavior for the intended reason.

An end-to-end failure is ambiguous. The source observation may omit the necessary information; the learned bridge may fail; the receiver-side insertion path may be incapable of expressing the task; or the functional evaluator may be insensitive or unsafe. Conversely, geometric success is also ambiguous. Low reconstruction error, high cosine similarity, or accurate task retrieval can coexist with negligible downstream causal effect.

This paper presents two registered studies from the LIP (Latent Injection Protocol) research program. Together they implement a receiver-anchored evaluation sequence:

1. validate the receiver carrier with receiver-native matched and identity-shuffled packets;
2. localize causally useful packet components while holding structural capacity constant;
3. train and select cross-model bridges without consulting sealed confirmation tasks;
4. compare learned matched packets with identity-shuffled, mean-scaffold, and norm-matched random controls under functional execution;
5. make claims at the task level and stop when a preregistered gate fails.

[[FIGURE:PIPELINE]]

The first study establishes that the selected receiver carrier can express task identity. A 24-position terminal packet copied from the receiver's own task-specific computation drives successful code generation, while an equally sized packet from another task does not. A source factorial further shows that the instruction core and function-name regions each contribute to the effect in an otherwise matched packet.

The second study asks whether a learned heterogeneous bridge can recreate that functionally useful state from a DeepSeek-Coder sender. Its development geometry is encouraging: three of three primary replicas pass joint, core, and name retrieval-margin gates. Its sealed functional result is not. Learned matched packets perform exactly like learned identity-shuffled packets at the task-clustered level.

The contribution is not the first proposal for latent communication. Earlier work communicates through activations, state trajectories, hidden states, and KV caches [1-6]. Our narrower contribution is a falsifiable evaluation pattern and a concrete result: a positive receiver-native oracle anchor can coexist with a zero learned identity effect, even after successful held-out geometric gates. This distinction matters when deciding whether a learned latent protocol has transported semantics or only organized representations.

# 2. Related work and novelty boundary

Ramesh and Li combine intermediate activations between language-model agents and report accuracy and efficiency gains [1]. State Delta Trajectory augments text communication with hidden-state changes [2]. Interlat communicates last hidden states and studies fully latent inter-agent exchange [3]. Cache-to-Cache learns to project and fuse source and target KV caches [4]. Subsequent work has investigated sparse cross-architecture alignment and direct cache translation [5,6]. A recent survey organizes this expanding area by representation, alignment, and receiver fusion [7].

These systems differ in whether the sender and receiver share an architecture, whether text remains present, which internal object is transmitted, and how function is measured. The present paper does not claim priority over latent communication as a general idea, over activation injection, or over cross-model representation mapping. It instead focuses on three evaluation features used together:

- a receiver-native oracle identity anchor before interpreting a learned bridge;
- matched-versus-shuffled interventions that preserve packet geometry and capacity;
- sealed, task-clustered functional confirmation that can invalidate a promising geometric development result.

The design is also related to causal abstraction and activation-intervention work [8,9]. Those traditions emphasize that representational association and causal effect are different objects. LIP applies that distinction to a communication channel: the packet must not only be decodable or close to a teacher state; its task identity must alter receiver behavior relative to identity-destroying controls.

# 3. Receiver-anchored evaluation

## 3.1 Two estimands, not one

Let task i have a receiver-native task residual packet D_i, an identity-shuffled donor packet D_pi(i), a learned packet Dhat_i predicted from the matching sender task, and a learned packet Dhat_pi(i) predicted from a same-stratum donor. Let Y(D) be a task-level functional outcome after receiver injection.

The receiver carrier effect is estimated by:

    tau_oracle = mean_i [ Y(D_i) - Y(D_pi(i)) ].

The learned identity effect is estimated by:

    tau_learned = mean_i [ Y(Dhat_i) - Y(Dhat_pi(i)) ].

A positive tau_oracle establishes that the selected receiver prompt, insertion path, packet geometry, and evaluation can express task identity on the tested population. It does not prove that a sender observation or learned bridge can reproduce the needed state. A positive learned effect requires the second contrast. If tau_oracle is positive and tau_learned is zero, the complete learned path failed despite a viable receiver carrier.

## 3.2 Identity controls

The principal negative control changes identity while preserving structure. Each shuffled packet comes from one other task in the same tokenizer-layout stratum. Donors are assigned by a derangement, so no task donates to itself. This control preserves packet shape, injected scalar count, insertion positions, insertion layers, and donor coherence.

The learned confirmation adds two controls. A mean scaffold contains only the training-set receiver mean. A random residual control samples an isotropic residual and rescales it to match the learned residual norm by layer. The four learned-side conditions therefore ask distinct questions:

- learned matched versus learned shuffled: does the learned packet carry the correct task identity?
- learned matched versus mean scaffold: does the learned residual add function beyond the shared scaffold?
- learned matched versus norm-matched random: is success attributable to learned direction rather than residual magnitude alone?

## 3.3 Functional and statistical units

Generated Python is scored by task tests inside a hardened, network-isolated Linux namespace. Syntax-only scoring is explicitly ineligible for a semantic-transport claim. The task is the statistical unit. Multiple generation seeds and, where applicable, bridge replicas are averaged within task before contrasts. Bootstrap intervals resample tasks. One-sided sign-flip tests operate on paired task differences, and Holm correction controls each registered family [10].

This prevents generation seeds, packet sites, or bridge replicas from being treated as independent task evidence. It also makes a zero identity effect interpretable: learned matched and shuffled conditions were compared on the same tasks under the same carrier and evaluator.

# 4. Study A: constant-capacity causal localization

## 4.1 Research question

The first study asks which components of a receiver-native terminal residual packet must preserve the target task's identity. It is an oracle replay study inside one pinned target model, not a learned inter-model bridge.

The population contains 32 MBPP test tasks [11], selected before latent confirmation by a text-only capability screen. Sixteen tasks have two-token function names and sixteen have three-token names. The receiver is NousResearch/Meta-Llama-3-8B-Instruct at revision 53346005fb0ef11d3b6a83b12c895cca40156b6c. Task states are captured from and replayed into the same receiver.

## 4.2 Packet and factorial

The packet spans 24 terminal prompt positions and the inputs to the first eight decoder blocks. Each packet is partitioned into:

- C, the terminal instruction core: 16 positions for two-token names or 15 for three-token names;
- N, the required function-name states: two or three positions;
- B, six fixed generation-boundary positions.

For each component, M denotes states matched to the target task and S denotes states from one same-stratum donor. The 2 x 2 x 2 conditions are MMM, SMM, MSM, MMS, SSM, SMS, MSS, and SSS. All contain exactly 24 populated positions. Two K=32 replication controls plus neutral and text controls yield 12 conditions. With 32 tasks and three fresh generation seeds, confirmation contains 1,152 registered records.

The ordered gates first test full-K=32 matched versus shuffled, then terminal-K=24 MMM versus SSS. Only after both gates reject does a Holm family of seven component contrasts open. Three contribution contrasts replace one matched component in MMM. Four rescue contrasts return selected target components to an otherwise donor packet.

## 4.3 Results

Both ordered gates passed. Full-K=32 matched replay achieved 85/96 functional passes (88.54%) versus 0/96 shuffled. Terminal-K=24 MMM achieved 79/96 (82.29%) versus 0/96 SSS. The corresponding mean task-level effects were 0.885417 and 0.822917, each with one-sided Monte Carlo p = 0.0000099999.

[[FIGURE:P013]]

| Condition | Packet identity | Passes | Rate |
| --- | --- | ---: | ---: |
| Neutral | no latent packet | 0/96 | 0.00% |
| Task text | textual task prompt | 87/96 | 90.62% |
| Full K=32 matched | matched full packet | 85/96 | 88.54% |
| Full K=32 shuffled | donor full packet | 0/96 | 0.00% |
| MMM | matched core, name, boundary | 79/96 | 82.29% |
| SMM | donor core | 3/96 | 3.12% |
| MSM | donor name | 0/96 | 0.00% |
| MMS | donor boundary | 78/96 | 81.25% |
| SMS | matched name only | 4/96 | 4.17% |
| MSS / SSM / SSS | remaining donor controls | 0/288 | 0.00% |

The seven-test family confirmed two contributions:

| Registered contrast | Mean difference | 95% task-bootstrap interval | Holm p | Result |
| --- | ---: | ---: | ---: | --- |
| Core contribution: MMM - SMM | 0.791667 | [0.645833, 0.916667] | 0.0000699993 | confirmed |
| Name contribution: MMM - MSM | 0.822917 | [0.687500, 0.937500] | 0.0000699993 | confirmed |
| Boundary contribution: MMM - MMS | 0.010417 | [0.000000, 0.031250] | 1.0 | not confirmed |
| Core-only rescue: MSS - SSS | 0.000000 | [0.000000, 0.000000] | 1.0 | not confirmed |
| Name-only rescue: SMS - SSS | 0.041667 | [0.000000, 0.114583] | 1.0 | not confirmed |
| Boundary-only rescue: SSM - SSS | 0.000000 | [0.000000, 0.000000] | 1.0 | not confirmed |
| Name-plus-boundary rescue: SMM - SSS | 0.031250 | [0.000000, 0.093750] | 1.0 | not confirmed |

The factorial surface is conjunctive. Replacing only function-name identity reduced observed success from 82.29% to 0%; replacing only core identity reduced it to 3.12%. Neither component alone rescued an otherwise donor packet. Replacing only boundary identity left 78/96 passes, one below MMM. Because equivalence and non-inferiority were not preregistered, this near-equality does not establish universal boundary irrelevance. It establishes only that the registered positive boundary-contribution contrast did not reject.

Descriptive geometry agrees with the intervention after the packet is decomposed. Common function-name positions had a mean task-signal fraction of 0.7050, compared with 0.5875 for common core positions and 0.0297 for boundary positions. Mean residual norms were comparable (2.3281, 2.2025, and 1.9974, respectively), illustrating why magnitude is not a task-information score.

# 5. Study B: learned heterogeneous transport

## 5.1 Sender, receiver, and packet bridge

The second study fixes a heterogeneous path from deepseek-ai/deepseek-coder-1.3b-base at revision e5babb80b8539a4e85dd2418c0ee611522276987 to the same pinned Llama-3-8B-Instruct receiver [12,13]. The sender observation has shape 24 layers x 32 terminal positions x 2048 features. Each sender state is projected to width 512, normalized, augmented with learned layer and position embeddings, and flattened into 768 source-memory sites.

Thirty-two learned protocol queries pass through two pre-normalized Transformer decoder blocks with eight-head cross-attention to produce a fixed 32 x 512 LIP code. The receiver decoder forms 8 x 24 receiver-site queries, cross-attends them to that code in two further blocks, and projects each decoded site to a normalized residual of width 4096. At inference, training-only site scales and the mean scaffold reconstruct the absolute packet as H_hat = mu_train + sigma * Delta_hat before injection into the residual inputs of receiver blocks 0 through 7.

[[FIGURE:BRIDGE]]

All splits are by task: 256 MBPP train tasks, 32 development-selection tasks, 32 disjoint development-gate tasks, and 32 sealed confirmation tasks. The confirmation cohort is balanced across the same two function-name strata and is disjoint from the preceding oracle-confirmation cohort and all bridge train/development tasks.

Three systems are trained with three registered seeds: a component-contrastive nonlinear bridge, the same nonlinear bridge with centered regression only, and a structured linear regression baseline. Checkpoint selection consults development-selection only. The selected checkpoint is opened once on development-gate. A primary replica passes only if joint, core, and name task-retrieval margins are positive under one Holm family. Confirmation opens only if at least two of three primary replicas pass.

All three component-contrastive replicas passed this gate. All three structured-linear replicas also passed; centered regression passed zero of three. The registered primary aggregate rule therefore opened confirmation. This is an important intermediate observation, but it is not a functional result.

## 5.2 Sealed functional confirmation

Confirmation contains 1,344 complete cells. Neutral, text-only, oracle-matched, oracle-shuffled, and mean-scaffold conditions each contain 96 generations. Learned matched, learned shuffled, and norm-matched random conditions each contain 288 generations because they cross 32 tasks, three bridge replicas, and three generation seeds.

The receiver-native oracle anchor passed again: oracle matched achieved 84/96 (87.50%), while oracle shuffled achieved 0/96. The carrier, receiver prompt, and task evaluator can therefore express the required identity on this cohort.

The learned bridge failed its primary functional family. Learned matched and learned shuffled each achieved 1/288 (0.3472%). After averaging replicas and generation seeds within task, their mean paired difference was exactly zero, their interval was [0, 0], and the one-sided p-value was 1.0.

[[FIGURE:P014]]

| Confirmation condition | Passes | Rate |
| --- | ---: | ---: |
| Neutral, no packet | 0/96 | 0.0000% |
| Text only | 89/96 | 92.7083% |
| Oracle teacher matched | 84/96 | 87.5000% |
| Oracle teacher shuffled | 0/96 | 0.0000% |
| Mean scaffold | 0/96 | 0.0000% |
| Learned matched | 1/288 | 0.3472% |
| Learned shuffled | 1/288 | 0.3472% |
| Random residual, norm matched | 0/288 | 0.0000% |

| Task-clustered contrast | Mean difference | 95% interval | Holm p | Confirmed |
| --- | ---: | ---: | ---: | --- |
| Oracle matched - oracle shuffled | 0.875000 | [0.760417, 0.968750] | n/a | yes |
| Learned matched - learned shuffled | 0.000000 | [0.000000, 0.000000] | 1.0 | no |
| Learned matched - mean scaffold | 0.003472 | [0.000000, 0.010417] | 1.0 | no |
| Learned matched - random norm matched | 0.003472 | [0.000000, 0.010417] | 1.0 | no |

The observed oracle identity effect was 0.875 and the learned identity effect was 0. The identity-recovery ratio was therefore 0.0. The final claim-eligible hardened summary reported semantic_transport_supported=false.

# 6. What the two studies establish

The positive oracle results show that task identity can be carried by a compact receiver-native residual packet under the tested Llama revision, prompt template, task population, and first-eight-block insertion path. The constant-capacity factorial identifies a distributed and conjunctive code: both instruction-core and function-name identities matter in an otherwise matched packet.

The heterogeneous result then separates representational organization from communication function. Held-out task retrieval and positive diagonal margins showed that the learned representations were geometrically task-discriminative. Those diagnostics did not predict a causal identity effect during generation. In this system, development separability was necessary for opening the registered test but was not sufficient for semantic transport.

The receiver-native anchor materially narrows the failure. A broken receiver carrier, incapable receiver, or globally nonfunctional evaluation path is inconsistent with 84/96 oracle-matched passes and 0/96 oracle-shuffled passes. The anchor does not identify one remaining cause. Failure may arise from information absent at the selected DeepSeek layers, the source observation, the scaffold/residual factorization, bridge capacity, optimization, objectives, or distribution shift between development geometry and executable behavior.

The scientific result is therefore two-part:

- positive, bounded evidence that receiver-native residual packets can causally transmit task identity and that constant-capacity identity interventions can localize useful subchannels;
- negative, bounded evidence that the registered DeepSeek-to-Llama bridge did not transport functionally detectable task identity despite passing its geometric development gates.

# 7. Limitations and prohibited inferences

The evidence is limited to code-generation tasks from MBPP, capability-screened populations, one pinned Llama receiver, one pinned DeepSeek sender, one prompt family, one terminal-packet geometry, and the tested bridge families and data budget. Capability screening improves assay sensitivity but changes the target population; rates should not be generalized to all MBPP tasks.

The 24-position packet is the smallest packet confirmed among the tested candidates. It is not a mathematical minimum. The absence of a confirmed positive boundary contribution is not an equivalence result and does not show that boundary states are irrelevant in other prompts, tasks, models, or interventions.

The negative learned result does not prove that heterogeneous latent transport is impossible. It rejects the complete registered system at its functional gate. Conversely, the positive oracle replay does not establish a learned protocol, arbitrary model-to-model composition, a model-independent interlingua, replacement of text in general, superior efficiency, or production readiness.

No claim of being the first latent-communication method is made. The priority claim of this manuscript is attached to the documented LIP evaluation protocol, its constant-capacity terminal-source experiment, and its receiver-anchored negative cross-model finding.

# 8. Reproducibility and artifact record

The public code repository is https://github.com/zigfreud/latent-llm-agent-communication. The paper-facing protocols are:

- docs/LIP-PROTO-013_terminal_source_factorial.md
- docs/LIP-PROTO-014_source_conditioned_residual_packet.md

They bind the task selectors, packet contracts, statistical gates, model revisions, generation grids, execution mode, and claim boundaries. The compact execution record is:

- Study A: 1,152 registered confirmation records, zero missing cells, and zero maximum self-replay logit error for K=32 and K=24.
- Study B: source commit 8e79a4740c9fd0fd98977c4b29feb3d28a45aa6e, 1,344 complete confirmation cells, and a passing independent hardened functional audit.
- Study B payload: 23 content-addressed scientific files; SHA256SUMS digest 2066ebe36d61da87e3d1c0f5dc8f8f6f2e93ca472e4405d783f0a7a64caa1db1.
- Hardened functional summary digest: 99e425d85bfc0aad3b03e0d42a90912d3789f273d7ba8ec25d354581c1897f38.

The repository includes a machine-readable result snapshot and a claim ledger beside this manuscript. Large model-derived tensors and checkpoints are not embedded in the paper. Their content hashes and validation reports bind the executed artifacts; an archival release should deposit redistributable claim-level artifacts with the manuscript and code snapshot.

# 9. Declarations and Disclosures

## Author Contributions

Cristiano Silva is the sole author and project lead. He is responsible for conceptualization, methodology, software, investigation, data curation, formal analysis, visualization, writing, and project administration.

## Use of Generative AI

Generative-AI tools, including OpenAI Codex and other LLM assistants, were used under the author's direct supervision as software-development, drafting, and editorial aids for implementation, documentation, and manuscript editing. The author reviewed and verified the scientific artifacts, code, analyses, and text and accepts full responsibility for the study design, reported evidence, interpretation, and conclusions.

## Ethics and Data Safety

This study evaluates the publicly available Mostly Basic Python Problems (MBPP) code-generation benchmark and does not involve human participants, personal data, or private repositories. All candidate-code execution used for claim-eligible functional evaluation was confined to a hardened, network-isolated Linux namespace.

## License

The manuscript text and figures are licensed under the Creative Commons Attribution 4.0 International license (CC BY 4.0). The accompanying codebase and protocol specifications are licensed separately under the MIT License.

# 10. Conclusion

Latent communication should be evaluated as a sequence of causal questions, not as one similarity score. A receiver-native oracle anchor establishes whether the target carrier can express identity. Identity shuffles then distinguish task-specific signal from packet capacity and magnitude. Sealed functional confirmation determines whether a learned bridge actually transports that identity.

Applied to LIP, this sequence yields a strong oracle result and a clear learned failure. The receiver can act on compact residual packets, and task identity depends on both instruction-core and function-name states. Yet a learned heterogeneous bridge that looked successful under held-out geometric diagnostics produced no functional identity effect. Reporting both outcomes defines a reproducible baseline for future bridge designs and prevents representational alignment from being mistaken for communication.

[[PAGEBREAK]]

# References

[1] V. Ramesh and K. Li. Communicating Activations Between Language Model Agents. Proceedings of the 42nd International Conference on Machine Learning, PMLR 267:51094-51116, 2025. https://proceedings.mlr.press/v267/ramesh25a.html

[2] Y. Tang, W. Su, Y. Zhou, Y. Liu, M. Zhang, S. Ma, and Q. Ai. Augmenting Multi-Agent Communication with State Delta Trajectory. arXiv:2506.19209, 2025. https://arxiv.org/abs/2506.19209

[3] Z. Du, R. Wang, H. Bai, Z. Cao, X. Zhu, B. Zheng, W. Chen, and H. Ying. Enabling Agents to Communicate Entirely in Latent Space. arXiv:2511.09149, 2025. https://arxiv.org/abs/2511.09149

[4] T. Fu, Z. Min, H. Zhang, J. Yan, G. Dai, W. Ouyang, and Y. Wang. Cache-to-Cache: Direct Semantic Communication Between Large Language Models. arXiv:2510.03215, 2025. https://arxiv.org/abs/2510.03215

[5] M. Wenzel. Latent Communication Between Language Model Agents: Channels, Alignment, and the Limits of Text. arXiv:2607.14103, 2026. https://arxiv.org/abs/2607.14103

[6] T. Heo, R. Shafipour, R. Zhao, M. Golub, M. M. Kamani, R. Borkar, M. T. Chandran, P. Zardoshti, and B. D. Rouhani. Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse. arXiv:2608.03893, 2026. https://arxiv.org/abs/2608.03893

[7] Y. Liu. Beyond Tokens: A Unified Framework for Latent Communication in LLM-Based Multi-Agent Systems. arXiv:2606.05711, 2026. https://arxiv.org/abs/2606.05711

[8] A. Geiger, H. Lu, T. Icard, and C. Potts. Causal Abstractions of Neural Networks. Advances in Neural Information Processing Systems, 34, 2021. https://arxiv.org/abs/2106.02997

[9] F. Zhang and N. Nanda. Towards Best Practices of Activation Patching in Language Models: Metrics and Methods. arXiv:2309.16042, 2023. https://arxiv.org/abs/2309.16042

[10] S. Holm. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics, 6(2):65-70, 1979.

[11] J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, and C. Sutton. Program Synthesis with Large Language Models. arXiv:2108.07732, 2021. https://arxiv.org/abs/2108.07732

[12] A. Grattafiori et al. The Llama 3 Herd of Models. arXiv:2407.21783, 2024. https://arxiv.org/abs/2407.21783

[13] D. Guo et al. DeepSeek-Coder: When the Large Language Model Meets Programming — The Rise of Code Intelligence. arXiv:2401.14196, 2024. https://arxiv.org/abs/2401.14196
