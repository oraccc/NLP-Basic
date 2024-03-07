# Research Papers



关于**Data Contamination**研究的不同方向

- **Detect** data contamination in (closed) LLMs

  - 使用各类方法去检测训练数据集闭源的大模型的数据集污染问题：如果数据集开源，可以检测重复率；如果数据集闭源，可以通过补全prompt的方式。


  > 1. Investigating Data Contamination in Modern Benchmarks for Large Language Models
  >
  > 2. Time Travel in LLMs: Tracing Data Contamination in Large Language Models
  >
  > 3. Proving Test Set Contamination in Black Box Language Models
  > 4. Detecting Pretraining Data from Large Language Models
  > 5. Did ChatGPT cheat on your test?
  > 6. Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models

- Practical strategies for **mitigating** data contamination

  - 如何使测试数据集保密，不对外公开进行测试，涉及可信通信的领域


  > 1. Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks
  > 2. Can we trust the evaluation on ChatGPT?

- Different **evaluation** strategies

  - 如何对已知有污染的模型进行更好地评测，动态检测

  > 1. DyVal: Graph-informed Dynamic Evaluation of Large Language Models
  > 1. Private Benchmarking to Prevent Contamination and Improve Comparative Evaluation of LLMs






## Investigating Data Contamination in Modern Benchmarks for Large Language Models

[Link](https://arxiv.org/pdf/2311.09783.pdf)

### Summary

The authors propose two methods to investigate data contamination in LLMs. 

First, they introduce a **retrieval-based system** to identify **overlaps** between evaluation benchmarks and pretraining data. 

Second, they propose a novel investigation protocol called **Testset Slot Guessing (TS-Guessing)** applicable to both open and proprietary models. This method involves masking a wrong answer in a multiple-choice question or obscuring an unlikely word in an evaluation example and prompting the model to fill in the gap. 



## Time Travel in LLMs: Tracing Data Contamination in Large Language Models

[Link](https://arxiv.org/pdf/2308.08493.pdf)

### Summary

The proposed approach starts by identifying potential contamination at the **instance level** and then extends this assessment to the **partition level**.

To detect contamination at the instance level, the method employs "guided instruction," prompting the LLM to complete a partial reference instance. If the LLM's output matches the latter segment of the reference, the instance is flagged as contaminated. 

To assess partition-level contamination, two approaches are proposed: 

* one based on average overlap scores with reference instances 
* another utilizing a classifier based on GPT-4 with few-shot in-context learning prompts. The method achieves high accuracy (between 92% and 100%) in detecting contamination across seven datasets, compared to manual evaluation by human experts.





## Proving Test Set Contamination in Black Box Language Models

[Link](https://arxiv.org/pdf/2310.17623.pdf)

### Summary

The paper presents a method to provide provable guarantees of test set contamination in language models without needing access to pretraining data or model weights. This approach relies on the principle that in the absence of contamination, **all orderings of an exchangeable benchmark should be equally likely**. However, contaminated language models tend to **favor certain canonical orderings** over others. The proposed test flags potential contamination whenever the likelihood of a canonically ordered benchmark dataset is significantly higher than the likelihood after shuffling the examples.





## Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks

[Link](https://arxiv.org/pdf/2305.10160.pdf)

### Summary

Strategies like leaderboards with hidden answers or using guaranteed unseen test data are costly and fragile over time. Assuming all relevant parties value clean test data and will collaborate to mitigate contamination, the abstract proposes three strategies:

* Encrypting public test data with a public key and licensing it to prevent derivative distribution

* Demanding training exclusion controls from closed API providers and refusing evaluation without them

* Avoiding data available with solutions online and releasing the webpage context of internet-derived data along with the data. 



## DyVal: Graph-informed Dynamic Evaluation of Large Language Models

[Link](https://arxiv.org/pdf/2309.17167.pdf)

### Summary

DyVal utilizes a dynamic evaluation framework to generate evaluation samples with varying complexities, leveraging directed acyclic graphs. These samples include challenging tasks in mathematics, logical reasoning, and algorithm problems. Evaluation of various LLMs, including Flan-T5-large, ChatGPT, and GPT4, demonstrates their performance variation across different complexities, highlighting the importance of dynamic evaluation. 



## Detecting Pretraining Data from Large Language Models

[Link](https://arxiv.org/pdf/2310.16789.pdf)

### Summary

The paper introduces a dynamic benchmark called WIKIMIA, which utilizes pre- and post-training data to support gold truth detection. Additionally, a new detection method named Min-K% Prob is proposed, which relies on the hypothesis that unseen examples are likely to contain outlier words with low probabilities under the LLM. This method does not require knowledge of the pretraining corpus or additional training, distinguishing it from previous detection methods.



## Can we trust the evaluation on ChatGPT?

[Link](https://arxiv.org/pdf/2303.12767.pdf)

### Summary

We highlight the issue of data contamination in ChatGPT evaluations, with a case study of the task of stance detection. We discuss the challenge of preventing data contamination and ensuring fair model evaluation in the age of closed and continuously trained models.



## Did ChatGPT cheat on your test?

[Link](https://hitz-zentroa.github.io/lm-contamination/blog/)



## Private Benchmarking to Prevent Contamination and Improve Comparative Evaluation of LLMs

[Link](https://arxiv.org/html/2403.00393v1)

### Summary

the authors propose "Private Benchmarking," a solution where test datasets are kept private, and models are evaluated without exposing the test data to the model. They outline different scenarios depending on the level of trust placed on model owners or dataset owners and present solutions to prevent data contamination using private benchmarking. Additionally, they discuss techniques from confidential computing and cryptography to maintain the privacy of model weights in private benchmarking scenarios. Finally, they offer solutions for auditing benchmark datasets to ensure their quality in private benchmarking settings.



## Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models

[Link](https://arxiv.org/html/2311.06233v5)

### Summary

The abstract introduces the Data Contamination Quiz (DCQ), a straightforward and efficient method for detecting data contamination in large language models (LLMs) and estimating its extent. The approach frames contamination detection as a series of multiple-choice questions, where three perturbed versions of each dataset instance are generated alongside the original. These versions differ only at the word level. The LLM is tasked with identifying the original instance from the provided options, with an additional option accounting for the possibility that none are correct. Since the sole distinguishing factor among the choices is the exact wording relative to the original instance, LLMs tend to gravitate towards the original if exposed to it during pretraining—an inherent trait of LLMs.