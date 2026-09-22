# KV Cache Compliace/Safety

## System-prompt based safety/compliance instructions are also evicted due to routine memory management through eviction.

A recent paper studies how eviction can impact policy/compliance degradation due to memory eviction and compaction. Since system prompts are usually where critical rules and compliance are stated, this has ramifications in how LLMs and Agentic workflows are deployed. Our study verifies the mechanical aspect of how eviction causes system-prompt context loss.

Our experiment shows that compliance drops sharply once a token retention drops below a certain threshold. The model starts answering and complying with forbidden requests stated in the system-prompts, but it doesn't affect RLHF based refusals (such as explicit harm/fraud requests). Our experiment is not based on jailbreak or adversarial pressure, but simply the mechanistic act of throwing away the system-prompt during eviction. The motivation of this study was that naive eviction remains an acknowledged risk (e.g., vLLM issue #36311).

<details>
  <summary><h3>Results</h3></summary>
  ### Baseline
  Our dataset includes two synthetic policies: a policy that forbids the LLM to answer questions regarding Ice Cream shops, and another policy forbids you to carry out certain DevOps operations like deleting, dropping, or modifying a database. Additionally, both policies have equivalent neutral or matched prompts.
  Under "no_eviction", the LLM follows both the policies diligently. For the "Ice Cream" scenario, the LLM refuses to recommend any nearby ice cream shops (100% refusal), while freely recommending coffee shops (neutral prompts, 0% refusal). For the "DevOps" operations, the model refuses 97% of the prompts that asks for deletion or drops, but in this case, also refuses 29% of the neutral prompts. This could be an instance where the models prior training is contributing to it being more cautious.
  This two baseline gives us comparison to test how eviction interferes.

  ### Compliance dropping sharply
  When we move our window size from "no_eviction" to "large" ("large" window size comfortable keeps the system-prompts in its context), we see that it produces no change. Compliance stays roughly 97-100% for both policies. But when we move from "large" to "medium" window size, we can see that token retention is roughly at 19-22% and compliance falls to 0% (ice cream) and 1% (devOps). The "small" window size doesn't show further collapse because collapse at "medium" is already at near zero.
  <table>
  <tr>
  <td><img src="/figures/fig1_compliance_collapse.png" width="400"></td>
  <td><img src="/figures/fig2_system_prompt_retention.png" width="400"></td>
  </tr>
  </table>
</details>
