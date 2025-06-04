# DSPy Educational Book


## Learn

### learn/evaluation/data.md

---
sidebar_position: 5
---

# Data

DSPy is a machine learning framework, so working in it involves training sets, development sets, and test sets. For each example in your data, we distinguish typically between three types of values: the inputs, the intermediate labels, and the final label. You can use DSPy effectively without any intermediate or final labels, but you will need at least a few example inputs.


## DSPy `Example` objects

The core data type for data in DSPy is `Example`. You will use **Examples** to represent items in your training set and test set. 

DSPy **Examples** are similar to Python `dict`s but have a few useful utilities. Your DSPy modules will return values of the type `Prediction`, which is a special sub-class of `Example`.

When you use DSPy, you will do a lot of evaluation and optimization runs. Your individual datapoints will be of type `Example`:

```python
qa_pair = dspy.Example(question="This is a question?", answer="This is an answer.")

print(qa_pair)
print(qa_pair.question)
print(qa_pair.answer)
```
**Output:**
```text
Example({'question': 'This is a question?', 'answer': 'This is an answer.'}) (input_keys=None)
This is a question?
This is an answer.
```

Examples can have any field keys and any value types, though usually values are strings.

```text
object = Example(field1=value1, field2=value2, field3=value3, ...)
```

You can now express your training set for example as:

```python
trainset = [dspy.Example(report="LONG REPORT 1", summary="short summary 1"), ...]
```


### Specifying Input Keys

In traditional ML, there are separated "inputs" and "labels".

In DSPy, the `Example` objects have a `with_inputs()` method, which can mark specific fields as inputs. (The rest are just metadata or labels.)

```python
# Single Input.
print(qa_pair.with_inputs("question"))

# Multiple Inputs; be careful about marking your labels as inputs unless you mean it.
print(qa_pair.with_inputs("question", "answer"))
```

Values can be accessed using the `.`(dot) operator. You can access the value of key `name` in defined object `Example(name="John Doe", job="sleep")` through `object.name`. 

To access or exclude certain keys, use `inputs()` and `labels()` methods to return new Example objects containing only input or non-input keys, respectively.

```python
article_summary = dspy.Example(article= "This is an article.", summary= "This is a summary.").with_inputs("article")

input_key_only = article_summary.inputs()
non_input_key_only = article_summary.labels()

print("Example object with Input fields only:", input_key_only)
print("Example object with Non-Input fields only:", non_input_key_only)
```

**Output**
```
Example object with Input fields only: Example({'article': 'This is an article.'}) (input_keys=None)
Example object with Non-Input fields only: Example({'summary': 'This is a summary.'}) (input_keys=None)
```

<!-- ## Loading Dataset from sources

One of the most convenient way to import datasets in DSPy is by using `DataLoader`. The first step is to declare an object, this object can then be used to call utilities to load datasets in different formats:

```python
from dspy.datasets import DataLoader

dl = DataLoader()
```

For most dataset formats, it's quite straightforward you pass the file path to the corresponding method of the format and you'll get the list of `Example` for the dataset in return:

```python
import pandas as pd

csv_dataset = dl.from_csv(
    "sample_dataset.csv",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

json_dataset = dl.from_json(
    "sample_dataset.json",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

parquet_dataset = dl.from_parquet(
    "sample_dataset.parquet",
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)

pandas_dataset = dl.from_pandas(
    pd.read_csv("sample_dataset.csv"),    # DataFrame
    fields=("instruction", "context", "response"),
    input_keys=("instruction", "context")
)
```

These are some formats that `DataLoader` supports to load from file directly. In the backend, most of these methods leverage the `load_dataset` method from `datasets` library to load these formats. But when working with text data you often use HuggingFace datasets, in order to import HF datasets in list of `Example` format we can use `from_huggingface` method:

```python
blog_alpaca = dl.from_huggingface(
    "intertwine-expel/expel-blog",
    input_keys=("title",)
)
```

You can access the splits of the dataset by accessing the corresponding key:

```python
train_split = blog_alpaca['train']

# Since this is the only split in the dataset we can split this into 
# train and test split ourselves by slicing or sampling 75 rows from the train
# split for testing.
testset = train_split[:75]
trainset = train_split[75:]
```

The way you load a HuggingFace dataset using `load_dataset` is exactly how you load data it via `from_huggingface` as well. This includes passing specific splits, subsplits, read instructions, etc. For code snippets, you can refer to the [cheatsheet snippets](/cheatsheet/#dspy-dataloaders) for loading from HF. -->

### learn/evaluation/metrics.md

---
sidebar_position: 5
---

# Metrics

DSPy is a machine learning framework, so you must think about your **automatic metrics** for evaluation (to track your progress) and optimization (so DSPy can make your programs more effective).


## What is a metric and how do I define a metric for my task?

A metric is just a function that will take examples from your data and the output of your system and return a score that quantifies how good the output is. What makes outputs from your system good or bad? 

For simple tasks, this could be just "accuracy" or "exact match" or "F1 score". This may be the case for simple classification or short-form QA tasks.

However, for most applications, your system will output long-form outputs. There, your metric should probably be a smaller DSPy program that checks multiple properties of the output (quite possibly using AI feedback from LMs).

Getting this right on the first try is unlikely, but you should start with something simple and iterate. 


## Simple metrics

A DSPy metric is just a function in Python that takes `example` (e.g., from your training or dev set) and the output `pred` from your DSPy program, and outputs a `float` (or `int` or `bool`) score.

Your metric should also accept an optional third argument called `trace`. You can ignore this for a moment, but it will enable some powerful tricks if you want to use your metric for optimization.

Here's a simple example of a metric that's comparing `example.answer` and `pred.answer`. This particular metric will return a `bool`.

```python
def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()
```

Some people find these utilities (built-in) convenient:

- `dspy.evaluate.metrics.answer_exact_match`
- `dspy.evaluate.metrics.answer_passage_match`

Your metrics could be more complex, e.g. check for multiple properties. The metric below will return a `float` if `trace is None` (i.e., if it's used for evaluation or optimization), and will return a `bool` otherwise (i.e., if it's used to bootstrap demonstrations).

```python
def validate_context_and_answer(example, pred, trace=None):
    # check the gold label and the predicted answer are the same
    answer_match = example.answer.lower() == pred.answer.lower()

    # check the predicted answer comes from one of the retrieved contexts
    context_match = any((pred.answer.lower() in c) for c in pred.context)

    if trace is None: # if we're doing evaluation or optimization
        return (answer_match + context_match) / 2.0
    else: # if we're doing bootstrapping, i.e. self-generating good demonstrations of each step
        return answer_match and context_match
```

Defining a good metric is an iterative process, so doing some initial evaluations and looking at your data and outputs is key.


## Evaluation

Once you have a metric, you can run evaluations in a simple Python loop.

```python
scores = []
for x in devset:
    pred = program(**x.inputs())
    score = metric(x, pred)
    scores.append(score)
```

If you need some utilities, you can also use the built-in `Evaluate` utility. It can help with things like parallel evaluation (multiple threads) or showing you a sample of inputs/outputs and the metric scores.

```python
from dspy.evaluate import Evaluate

# Set up the evaluator, which can be re-used in your code.
evaluator = Evaluate(devset=YOUR_DEVSET, num_threads=1, display_progress=True, display_table=5)

# Launch evaluation.
evaluator(YOUR_PROGRAM, metric=YOUR_METRIC)
```


## Intermediate: Using AI feedback for your metric

For most applications, your system will output long-form outputs, so your metric should check multiple dimensions of the output using AI feedback from LMs.

This simple signature could come in handy.

```python
# Define the signature for automatic assessments.
class Assess(dspy.Signature):
    """Assess the quality of a tweet along the specified dimension."""

    assessed_text = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer: bool = dspy.OutputField()
```

For example, below is a simple metric that checks a generated tweet (1) answers a given question correctly and (2) whether it's also engaging. We also check that (3) `len(tweet) <= 280` characters.

```python
def metric(gold, pred, trace=None):
    question, answer, tweet = gold.question, gold.answer, pred.output

    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
    correct = f"The text should answer `{question}` with `{answer}`. Does the assessed text contain this answer?"
    
    correct =  dspy.Predict(Assess)(assessed_text=tweet, assessment_question=correct)
    engaging = dspy.Predict(Assess)(assessed_text=tweet, assessment_question=engaging)

    correct, engaging = [m.assessment_answer for m in [correct, engaging]]
    score = (correct + engaging) if correct and (len(tweet) <= 280) else 0

    if trace is not None: return score >= 2
    return score / 2.0
```

When compiling, `trace is not None`, and we want to be strict about judging things, so we will only return `True` if `score >= 2`. Otherwise, we return a score out of 1.0 (i.e., `score / 2.0`).


## Advanced: Using a DSPy program as your metric

If your metric is itself a DSPy program, one of the most powerful ways to iterate is to compile (optimize) your metric itself. That's usually easy because the output of the metric is usually a simple value (e.g., a score out of 5) so the metric's metric is easy to define and optimize by collecting a few examples.



### Advanced: Accessing the `trace`

When your metric is used during evaluation runs, DSPy will not try to track the steps of your program.

But during compiling (optimization), DSPy will trace your LM calls. The trace will contain inputs/outputs to each DSPy predictor and you can leverage that to validate intermediate steps for optimization.


```python
def validate_hops(example, pred, trace=None):
    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True
```


### learn/evaluation/overview.md

---
sidebar_position: 1
---

# Evaluation in DSPy

Once you have an initial system, it's time to **collect an initial development set** so you can refine it more systematically. Even 20 input examples of your task can be useful, though 200 goes a long way. Depending on your _metric_, you either just need inputs and no labels at all, or you need inputs and the _final_ outputs of your system. (You almost never need labels for the intermediate steps in your program in DSPy.) You can probably find datasets that are adjacent to your task on, say, HuggingFace datasets or in a naturally occuring source like StackExchange. If there's data whose licenses are permissive enough, we suggest you use them. Otherwise, you can label a few examples by hand or start deploying a demo of your system and collect initial data that way.

Next, you should **define your DSPy metric**. What makes outputs from your system good or bad? Invest in defining metrics and improving them incrementally over time; it's hard to consistently improve what you aren't able to define. A metric is a function that takes examples from your data and takes the output of your system, and returns a score. For simple tasks, this could be just "accuracy", e.g. for simple classification or short-form QA tasks. For most applications, your system will produce long-form outputs, so your metric will be a smaller DSPy program that checks multiple properties of the output. Getting this right on the first try is unlikely: start with something simple and iterate.

Now that you have some data and a metric, run development evaluations on your pipeline designs to understand their tradeoffs. Look at the outputs and the metric scores. This will probably allow you to spot any major issues, and it will define a baseline for your next steps.


??? "If your metric is itself a DSPy program..."
    If your metric is itself a DSPy program, a powerful way to iterate is to optimize your metric itself. That's usually easy because the output of the metric is usually a simple value (e.g., a score out of 5), so the metric's metric is easy to define and optimize by collecting a few examples.



### learn/index.md

---
sidebar_position: 1
---

# Learning DSPy: An Overview

DSPy exposes a very small API that you can learn quickly. However, building a new AI system is a more open-ended journey of iterative development, in which you compose the tools and design patterns of DSPy to optimize for _your_ objectives. The three stages of building AI systems in DSPy are:

1) **DSPy Programming.** This is about defining your task, its constraints, exploring a few examples, and using that to inform your initial pipeline design.

2) **DSPy Evaluation.** Once your system starts working, this is the stage where you collect an initial development set, define your DSPy metric, and use these to iterate on your system more systematically.

3) **DSPy Optimization.** Once you have a way to evaluate your system, you use DSPy optimizers to tune the prompts or weights in your program.

We suggest learning and applying DSPy in this order. For example, it's unproductive to launch optimization runs using a poorly-design program or a bad metric.

### learn/optimization/optimizers.md

---
sidebar_position: 1
---

# DSPy Optimizers (formerly Teleprompters)


A **DSPy optimizer** is an algorithm that can tune the parameters of a DSPy program (i.e., the prompts and/or the LM weights) to maximize the metrics you specify, like accuracy.


A typical DSPy optimizer takes three things:

- Your **DSPy program**. This may be a single module (e.g., `dspy.Predict`) or a complex multi-module program.

- Your **metric**. This is a function that evaluates the output of your program, and assigns it a score (higher is better).

- A few **training inputs**. This may be very small (i.e., only 5 or 10 examples) and incomplete (only inputs to your program, without any labels).

If you happen to have a lot of data, DSPy can leverage that. But you can start small and get strong results.

**Note:** Formerly called teleprompters. We are making an official name update, which will be reflected throughout the library and documentation.


## What does a DSPy Optimizer tune? How does it tune them?

Different optimizers in DSPy will tune your program's quality by **synthesizing good few-shot examples** for every module, like `dspy.BootstrapRS`,<sup>[1](https://arxiv.org/abs/2310.03714)</sup> **proposing and intelligently exploring better natural-language instructions** for every prompt, like `dspy.MIPROv2`,<sup>[2](https://arxiv.org/abs/2406.11695)</sup> and **building datasets for your modules and using them to finetune the LM weights** in your system, like `dspy.BootstrapFinetune`.<sup>[3](https://arxiv.org/abs/2407.10930)</sup>

??? "What's an example of a DSPy optimizer? How do different optimizers work?"

    Take the `dspy.MIPROv2` optimizer as an example. First, MIPRO starts with the **bootstrapping stage**. It takes your program, which may be unoptimized at this point, and runs it many times across different inputs to collect traces of input/output behavior for each one of your modules. It filters these traces to keep only those that appear in trajectories scored highly by your metric. Second, MIPRO enters its **grounded proposal stage**. It previews your DSPy program's code, your data, and traces from running your program, and uses them to draft many potential instructions for every prompt in your program. Third, MIPRO launches the **discrete search stage**. It samples mini-batches from your training set, proposes a combination of instructions and traces to use for constructing every prompt in the pipeline, and evaluates the candidate program on the mini-batch. Using the resulting score, MIPRO updates a surrogate model that helps the proposals get better over time.

    One thing that makes DSPy optimizers so powerful is that they can be composed. You can run `dspy.MIPROv2` and use the produced program as an input to `dspy.MIPROv2` again or, say, to `dspy.BootstrapFinetune` to get better results. This is partly the essence of `dspy.BetterTogether`. Alternatively, you can run the optimizer and then extract the top-5 candidate programs and build a `dspy.Ensemble` of them. This allows you to scale _inference-time compute_ (e.g., ensembles) as well as DSPy's unique _pre-inference time compute_ (i.e., optimization budget) in highly systematic ways.



## What DSPy Optimizers are currently available?

Optimizers can be accessed via `from dspy.teleprompt import *`.

### Automatic Few-Shot Learning

These optimizers extend the signature by automatically generating and including **optimized** examples within the prompt sent to the model, implementing few-shot learning.

1. [**`LabeledFewShot`**](/api/optimizers/LabeledFewShot): Simply constructs few-shot examples (demos) from provided labeled input and output data points.  Requires `k` (number of examples for the prompt) and `trainset` to randomly select `k` examples from.

2. [**`BootstrapFewShot`**](/api/optimizers/BootstrapFewshot): Uses a `teacher` module (which defaults to your program) to generate complete demonstrations for every stage of your program, along with labeled examples in `trainset`. Parameters include `max_labeled_demos` (the number of demonstrations randomly selected from the `trainset`) and `max_bootstrapped_demos` (the number of additional examples generated by the `teacher`). The bootstrapping process employs the metric to validate demonstrations, including only those that pass the metric in the "compiled" prompt. Advanced: Supports using a `teacher` program that is a *different* DSPy program that has compatible structure, for harder tasks.

3. [**`BootstrapFewShotWithRandomSearch`**](/api/optimizers/BootstrapFewshotWithRandomSearch): Applies `BootstrapFewShot` several times with random search over generated demonstrations, and selects the best program over the optimization. Parameters mirror those of `BootstrapFewShot`, with the addition of `num_candidate_programs`, which specifies the number of random programs evaluated over the optimization, including candidates of the uncompiled program, `LabeledFewShot` optimized program, `BootstrapFewShot` compiled program with unshuffled examples and `num_candidate_programs` of `BootstrapFewShot` compiled programs with randomized example sets.

4. [**`KNNFewShot`**](/api/optimizers/KNNFewShot/). Uses k-Nearest Neighbors algorithm to find the nearest training example demonstrations for a given input example. These nearest neighbor demonstrations are then used as the trainset for the BootstrapFewShot optimization process.


### Automatic Instruction Optimization

These optimizers produce optimal instructions for the prompt and, in the case of MIPROv2 can also optimize the set of few-shot demonstrations.

5. [**`COPRO`**](/api/optimizers/COPRO): Generates and refines new instructions for each step, and optimizes them with coordinate ascent (hill-climbing using the metric function and the `trainset`). Parameters include `depth` which is the number of iterations of prompt improvement the optimizer runs over.

6. [**`MIPROv2`**](/api/optimizers/MIPROv2): Generates instructions *and* few-shot examples in each step. The instruction generation is data-aware and demonstration-aware. Uses Bayesian Optimization to effectively search over the space of generation instructions/demonstrations across your modules.


### Automatic Finetuning

This optimizer is used to fine-tune the underlying LLM(s).

7. [**`BootstrapFinetune`**](/api/optimizers/BootstrapFinetune): Distills a prompt-based DSPy program into weight updates. The output is a DSPy program that has the same steps, but where each step is conducted by a finetuned model instead of a prompted LM.


### Program Transformations

8. [**`Ensemble`**](/api/optimizers/Ensemble): Ensembles a set of DSPy programs and either uses the full set or randomly samples a subset into a single program.


## Which optimizer should I use?

Ultimately, finding the ‘right’ optimizer to use & the best configuration for your task will require experimentation. Success in DSPy is still an iterative process - getting the best performance on your task will require you to explore and iterate.  

That being said, here's the general guidance on getting started:

- If you have **very few examples** (around 10), start with `BootstrapFewShot`.
- If you have **more data** (50 examples or more), try  `BootstrapFewShotWithRandomSearch`.
- If you prefer to do **instruction optimization only** (i.e. you want to keep your prompt 0-shot), use `MIPROv2` [configured for 0-shot optimization to optimize](/deep-dive/optimizers/miprov2#optimizing-instructions-only-with-miprov2-0-shot). 
- If you’re willing to use more inference calls to perform **longer optimization runs** (e.g. 40 trials or more), and have enough data (e.g. 200 examples or more to prevent overfitting) then try `MIPROv2`. 
- If you have been able to use one of these with a large LM (e.g., 7B parameters or above) and need a very **efficient program**, finetune a small LM for your task with `BootstrapFinetune`.

## How do I use an optimizer?

They all share this general interface, with some differences in the keyword arguments (hyperparameters). Detailed documentation for key optimizers can be found [here](/deep-dive/optimizers/vfrs), and a full list can be found [here](https://dspy.ai/api/optimizers/BetterTogether/).

Let's see this with the most common one, `BootstrapFewShotWithRandomSearch`.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 8-shot examples of your program's steps.
# The optimizer will repeat this 10 times (plus some initial attempts) before selecting its best attempt on the devset.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, num_candidate_programs=10, num_threads=4)

teleprompter = BootstrapFewShotWithRandomSearch(metric=YOUR_METRIC_HERE, **config)
optimized_program = teleprompter.compile(YOUR_PROGRAM_HERE, trainset=YOUR_TRAINSET_HERE)
```


!!! info "Getting Started III: Optimizing the LM prompts or weights in DSPy programs"
    A typical simple optimization run costs on the order of $2 USD and takes around ten minutes, but be careful when running optimizers with very large LMs or very large datasets.
    Optimizer runs can cost as little as a few cents or up to tens of dollars, depending on your LM, dataset, and configuration.
    
    === "Optimizing prompts for a ReAct agent"
        This is a minimal but fully runnable example of setting up a `dspy.ReAct` agent that answers questions via
        search from Wikipedia and then optimizing it using `dspy.MIPROv2` in the cheap `light` mode on 500
        question-answer pairs sampled from the `HotPotQA` dataset.

        ```python linenums="1"
        import dspy
        from dspy.datasets import HotPotQA

        dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

        def search(query: str) -> list[str]:
            """Retrieves abstracts from Wikipedia."""
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]

        trainset = [x.with_inputs('question') for x in HotPotQA(train_seed=2024, train_size=500).train]
        react = dspy.ReAct("question -> answer", tools=[search])

        tp = dspy.MIPROv2(metric=dspy.evaluate.answer_exact_match, auto="light", num_threads=24)
        optimized_react = tp.compile(react, trainset=trainset)
        ```

        An informal run similar to this on DSPy 2.5.29 raises ReAct's score from 24% to 51%.

    === "Optimizing prompts for RAG"
        Given a retrieval index to `search`, your favorite `dspy.LM`, and a small `trainset` of questions and ground-truth responses, the following code snippet can optimize your RAG system with long outputs against the built-in `dspy.SemanticF1` metric, which is implemented as a DSPy module.

        ```python linenums="1"
        class RAG(dspy.Module):
            def __init__(self, num_docs=5):
                self.num_docs = num_docs
                self.respond = dspy.ChainOfThought('context, question -> response')

            def forward(self, question):
                context = search(question, k=self.num_docs)   # not defined in this snippet, see link above
                return self.respond(context=context, question=question)

        tp = dspy.MIPROv2(metric=dspy.SemanticF1(), auto="medium", num_threads=24)
        optimized_rag = tp.compile(RAG(), trainset=trainset, max_bootstrapped_demos=2, max_labeled_demos=2)
        ```

        For a complete RAG example that you can run, start this [tutorial](/tutorials/rag/). It improves the quality of a RAG system over a subset of StackExchange communities from 53% to 61%.

    === "Optimizing weights for Classification"
        This is a minimal but fully runnable example of setting up a `dspy.ChainOfThought` module that classifies
        short texts into one of 77 banking labels and then using `dspy.BootstrapFinetune` with 2000 text-label pairs
        from the `Banking77` to finetune the weights of GPT-4o-mini for this task. We use the variant
        `dspy.ChainOfThoughtWithHint`, which takes an optional `hint` at bootstrapping time, to maximize the utility of
        the training data. Naturally, hints are not available at test time. More can be found in this [tutorial](/tutorials/classification_finetuning/).

        <details><summary>Click to show dataset setup code.</summary>

        ```python linenums="1"
        import random
        from typing import Literal
        from dspy.datasets import DataLoader
        from datasets import load_dataset

        # Load the Banking77 dataset.
        CLASSES = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True).features['label'].names
        kwargs = dict(fields=("text", "label"), input_keys=("text",), split="train", trust_remote_code=True)

        # Load the first 2000 examples from the dataset, and assign a hint to each *training* example.
        trainset = [
            dspy.Example(x, hint=CLASSES[x.label], label=CLASSES[x.label]).with_inputs("text", "hint")
            for x in DataLoader().from_huggingface(dataset_name="PolyAI/banking77", **kwargs)[:2000]
        ]
        random.Random(0).shuffle(trainset)
        ```
        </details>

        ```python linenums="1"
        import dspy
        dspy.configure(lm=dspy.LM('gpt-4o-mini-2024-07-18'))
        
        # Define the DSPy module for classification. It will use the hint at training time, if available.
        signature = dspy.Signature("text -> label").with_updated_fields('label', type_=Literal[tuple(CLASSES)])
        classify = dspy.ChainOfThoughtWithHint(signature)

        # Optimize via BootstrapFinetune.
        optimizer = dspy.BootstrapFinetune(metric=(lambda x, y, trace=None: x.label == y.label), num_threads=24)
        optimized = optimizer.compile(classify, trainset=trainset)

        optimized(text="What does a pending cash withdrawal mean?")
        ```

        **Possible Output (from the last line):**
        ```text
        Prediction(
            reasoning='A pending cash withdrawal indicates that a request to withdraw cash has been initiated but has not yet been completed or processed. This status means that the transaction is still in progress and the funds have not yet been deducted from the account or made available to the user.',
            label='pending_cash_withdrawal'
        )
        ```

        An informal run similar to this on DSPy 2.5.29 raises GPT-4o-mini's score 66% to 87%.


## Saving and loading optimizer output

After running a program through an optimizer, it's useful to also save it. At a later point, a program can be loaded from a file and used for inference. For this, the `load` and `save` methods can be used.

```python
optimized_program.save(YOUR_SAVE_PATH)
```

The resulting file is in plain-text JSON format. It contains all the parameters and steps in the source program. You can always read it and see what the optimizer generated. You can add `save_field_meta` to additionally save the list of fields with the keys, `name`, `field_type`, `description`, and `prefix` with: `optimized_program.save(YOUR_SAVE_PATH, save_field_meta=True).


To load a program from a file, you can instantiate an object from that class and then call the load method on it.

```python
loaded_program = YOUR_PROGRAM_CLASS()
loaded_program.load(path=YOUR_SAVE_PATH)
```



### learn/optimization/overview.md

---
sidebar_position: 1
---


# Optimization in DSPy

Once you have a system and a way to evaluate it, you can use DSPy optimizers to tune the prompts or weights in your program. Now it's useful to expand your data collection effort into building a training set and a held-out test set, in addition to the development set you've been using for exploration. For the training set (and its subset, validation set), you can often get substantial value out of 30 examples, but aim for at least 300 examples. Some optimizers accept a `trainset` only. Others ask for a `trainset` and a `valset`. For prompt optimizers, we suggest starting with a 20% split for training and 80% for validation, which is often the _opposite_ of what one does for DNNs.

After your first few optimization runs, you are either very happy with everything or you've made a lot of progress but you don't like something about the final program or the metric. At this point, go back to step 1 (Programming in DSPy) and revisit the major questions. Did you define your task well? Do you need to collect (or find online) more data for your problem? Do you want to update your metric? And do you want to use a more sophisticated optimizer? Do you need to consider advanced features like DSPy Assertions? Or, perhaps most importantly, do you want to add some more complexity or steps in your DSPy program itself? Do you want to use multiple optimizers in a sequence?

Iterative development is key. DSPy gives you the pieces to do that incrementally: iterating on your data, your program structure, your assertions, your metric, and your optimization steps. Optimizing complex LM programs is an entirely new paradigm that only exists in DSPy at the time of writing (update: there are now numerous DSPy extension frameworks, so this part is no longer true :-), so naturally the norms around what to do are still emerging. If you need help, we recently created a [Discord server](https://discord.gg/XCGy2WDCQB) for the community.



### learn/programming/7-assertions.md

# DSPy Assertions 

!!! warning "This page is outdated and may not be fully accurate in DSPy 2.5"


## Introduction

Language models (LMs) have transformed how we interact with machine learning, offering vast capabilities in natural language understanding and generation. However, ensuring these models adhere to domain-specific constraints remains a challenge. Despite the growth of techniques like fine-tuning or “prompt engineering”, these approaches are extremely tedious and rely on heavy, manual hand-waving to guide the LMs in adhering to specific constraints. Even DSPy's modularity of programming prompting pipelines lacks mechanisms to effectively and automatically enforce these constraints. 

To address this, we introduce DSPy Assertions, a feature within the DSPy framework designed to automate the enforcement of computational constraints on LMs. DSPy Assertions empower developers to guide LMs towards desired outcomes with minimal manual intervention, enhancing the reliability, predictability, and correctness of LM outputs.

### dspy.Assert and dspy.Suggest API    

We introduce two primary constructs within DSPy Assertions:

- **`dspy.Assert`**:
  - **Parameters**: 
    - `constraint (bool)`: Outcome of Python-defined boolean validation check.
    - `msg (Optional[str])`: User-defined error message providing feedback or correction guidance.
    - `backtrack (Optional[module])`: Specifies target module for retry attempts upon constraint failure. The default backtracking module is the last module before the assertion.
  - **Behavior**: Initiates retry  upon failure, dynamically adjusting the pipeline's execution. If failures persist, it halts execution and raises a `dspy.AssertionError`.

- **`dspy.Suggest`**:
  - **Parameters**: Similar to `dspy.Assert`.
  - **Behavior**: Encourages self-refinement through retries without enforcing hard stops. Logs failures after maximum backtracking attempts and continues execution.

- **dspy.Assert vs. Python Assertions**: Unlike conventional Python `assert` statements that terminate the program upon failure, `dspy.Assert` conducts a sophisticated retry mechanism, allowing the pipeline to adjust. 

Specifically, when a constraint is not met:

- Backtracking Mechanism: An under-the-hood backtracking is initiated, offering the model a chance to self-refine and proceed, which is done through signature modification.
- Dynamic Signature Modification: internally modifying your DSPy program’s Signature by adding the following fields:
    - Past Output: your model's past output that did not pass the validation_fn
    - Instruction: your user-defined feedback message on what went wrong and what possibly to fix

If the error continues past the `max_backtracking_attempts`, then `dspy.Assert` will halt the pipeline execution, alerting you with an `dspy.AssertionError`. This ensures your program doesn't continue executing with “bad” LM behavior and immediately highlights sample failure outputs for user assessment.

- **dspy.Suggest vs. dspy.Assert**: `dspy.Suggest` on the other hand offers a softer approach. It maintains the same retry backtracking as `dspy.Assert` but instead serves as a gentle nudger. If the model outputs cannot pass the model constraints after the `max_backtracking_attempts`, `dspy.Suggest` will log the persistent failure and continue execution of the program on the rest of the data. This ensures the LM pipeline works in a "best-effort" manner without halting execution. 

- **`dspy.Suggest`** statements are best utilized as "helpers" during the evaluation phase, offering guidance and potential corrections without halting the pipeline.
- **`dspy.Assert`** statements are recommended during the development stage as "checkers" to ensure the LM behaves as expected, providing a robust mechanism for identifying and addressing errors early in the development cycle.


## Use Case: Including Assertions in DSPy Programs

We start with using an example of a multi-hop QA SimplifiedBaleen pipeline as defined in the intro walkthrough. 

```python
class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            prev_queries.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        
        pred = self.generate_answer(context=context, question=question)
        pred = dspy.Prediction(context=context, answer=pred.answer)
        return pred

baleen = SimplifiedBaleen()

baleen(question = "Which award did Gary Zukav's first book receive?")
```

To include DSPy Assertions, we simply define our validation functions and declare our assertions following the respective model generation. 

For this use case, suppose we want to impose the following constraints:
    1. Length - each query should be less than 100 characters
    2. Uniqueness - each generated query should differ from previously-generated queries. 
    
We can define these validation checks as boolean functions:

```python
#simplistic boolean check for query length
len(query) <= 100

#Python function for validating distinct queries
def validate_query_distinction_local(previous_queries, query):
    """check if query is distinct from previous queries"""
    if previous_queries == []:
        return True
    if dspy.evaluate.answer_exact_match_str(query, previous_queries, frac=0.8):
        return False
    return True
```

We can declare these validation checks through `dspy.Suggest` statements (as we want to test the program in a best-effort demonstration). We want to keep these after the query generation `query = self.generate_query[hop](context=context, question=question).query`.

```python
dspy.Suggest(
    len(query) <= 100,
    "Query should be short and less than 100 characters",
    target_module=self.generate_query
)

dspy.Suggest(
    validate_query_distinction_local(prev_queries, query),
    "Query should be distinct from: "
    + "; ".join(f"{i+1}) {q}" for i, q in enumerate(prev_queries)),
    target_module=self.generate_query
)
```

It is recommended to define a program with assertions separately than your original program if you are doing comparative evaluation for the effect of assertions. If not, feel free to set Assertions away!

Let's take a look at how the SimplifiedBaleen program will look with Assertions included:

```python
class SimplifiedBaleenAssertions(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query

            dspy.Suggest(
                len(query) <= 100,
                "Query should be short and less than 100 characters",
                target_module=self.generate_query
            )

            dspy.Suggest(
                validate_query_distinction_local(prev_queries, query),
                "Query should be distinct from: "
                + "; ".join(f"{i+1}) {q}" for i, q in enumerate(prev_queries)),
                target_module=self.generate_query
            )

            prev_queries.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        
        if all_queries_distinct(prev_queries):
            self.passed_suggestions += 1

        pred = self.generate_answer(context=context, question=question)
        pred = dspy.Prediction(context=context, answer=pred.answer)
        return pred
```

Now calling programs with DSPy Assertions requires one last step, and that is transforming the program to wrap it with internal assertions backtracking and Retry logic. 

```python
from dspy.primitives.assertions import assert_transform_module, backtrack_handler

baleen_with_assertions = assert_transform_module(SimplifiedBaleenAssertions(), backtrack_handler)

# backtrack_handler is parameterized over a few settings for the backtracking mechanism
# To change the number of max retry attempts, you can do
baleen_with_assertions_retry_once = assert_transform_module(SimplifiedBaleenAssertions(), 
    functools.partial(backtrack_handler, max_backtracks=1))
```

Alternatively, you can also directly call `activate_assertions` on the program with `dspy.Assert/Suggest` statements using the default backtracking mechanism (`max_backtracks=2`):

```python
baleen_with_assertions = SimplifiedBaleenAssertions().activate_assertions()
```

Now let's take a look at the internal LM backtracking by inspecting the history of the LM query generations. Here we see that when a query fails to pass the validation check of being less than 100 characters, its internal `GenerateSearchQuery` signature is dynamically modified during the backtracking+Retry process to include the past query and the corresponding user-defined instruction: `"Query should be short and less than 100 characters"`.


```text
Write a simple search query that will help answer a complex question.

---

Follow the following format.

Context: may contain relevant facts

Question: ${question}

Reasoning: Let's think step by step in order to ${produce the query}. We ...

Query: ${query}

---

Context:
[1] «Kerry Condon | Kerry Condon (born 4 January 1983) is [...]»
[2] «Corona Riccardo | Corona Riccardo (c. 1878October 15, 1917) was [...]»

Question: Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet." ?

Reasoning: Let's think step by step in order to find the answer to this question. First, we need to identify the actress who played Ophelia in a Royal Shakespeare Company production of "Hamlet." Then, we need to find out if this actress also acted in the short film "The Shore."

Query: "actress who played Ophelia in Royal Shakespeare Company production of Hamlet" + "actress in short film The Shore"



Write a simple search query that will help answer a complex question.

---

Follow the following format.

Context: may contain relevant facts

Question: ${question}

Past Query: past output with errors

Instructions: Some instructions you must satisfy

Query: ${query}

---

Context:
[1] «Kerry Condon | Kerry Condon (born 4 January 1983) is an Irish television and film actress, best known for her role as Octavia of the Julii in the HBO/BBC series "Rome," as Stacey Ehrmantraut in AMC's "Better Call Saul" and as the voice of F.R.I.D.A.Y. in various films in the Marvel Cinematic Universe. She is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet."»
[2] «Corona Riccardo | Corona Riccardo (c. 1878October 15, 1917) was an Italian born American actress who had a brief Broadway stage career before leaving to become a wife and mother. Born in Naples she came to acting in 1894 playing a Mexican girl in a play at the Empire Theatre. Wilson Barrett engaged her for a role in his play "The Sign of the Cross" which he took on tour of the United States. Riccardo played the role of Ancaria and later played Berenice in the same play. Robert B. Mantell in 1898 who struck by her beauty also cast her in two Shakespeare plays, "Romeo and Juliet" and "Othello". Author Lewis Strang writing in 1899 said Riccardo was the most promising actress in America at the time. Towards the end of 1898 Mantell chose her for another Shakespeare part, Ophelia im Hamlet. Afterwards she was due to join Augustin Daly's Theatre Company but Daly died in 1899. In 1899 she gained her biggest fame by playing Iras in the first stage production of Ben-Hur.»

Question: Who acted in the shot film The Shore and is also the youngest actress ever to play Ophelia in a Royal Shakespeare Company production of "Hamlet." ?

Past Query: "actress who played Ophelia in Royal Shakespeare Company production of Hamlet" + "actress in short film The Shore"

Instructions: Query should be short and less than 100 characters

Query: "actress Ophelia RSC Hamlet" + "actress The Shore"

```


## Assertion-Driven Optimizations

DSPy Assertions work with optimizations that DSPy offers, particularly with `BootstrapFewShotWithRandomSearch`, including the following settings:

- Compilation with Assertions
    This includes assertion-driven example bootstrapping and counterexample bootstrapping during compilation. The teacher model for bootstrapping few-shot demonstrations can make use of DSPy Assertions to offer robust bootstrapped examples for the student model to learn from during inference. In this setting, the student model does not perform assertion aware optimizations (backtracking and retry) during inference.
- Compilation + Inference with Assertions
    -This includes assertion-driven optimizations in both compilation and inference. Now the teacher model offers assertion-driven examples but the student can further optimize with assertions of its own during inference time. 
```python
teleprompter = BootstrapFewShotWithRandomSearch(
    metric=validate_context_and_answer_and_hops,
    max_bootstrapped_demos=max_bootstrapped_demos,
    num_candidate_programs=6,
)

#Compilation with Assertions
compiled_with_assertions_baleen = teleprompter.compile(student = baleen, teacher = baleen_with_assertions, trainset = trainset, valset = devset)

#Compilation + Inference with Assertions
compiled_baleen_with_assertions = teleprompter.compile(student=baleen_with_assertions, teacher = baleen_with_assertions, trainset=trainset, valset=devset)

```

### learn/programming/language_models.md

---
sidebar_position: 2
---

# Language Models

The first step in any DSPy code is to set up your language model. For example, you can configure OpenAI's GPT-4o-mini as your default LM as follows.

```python linenums="1"
# Authenticate via `OPENAI_API_KEY` env: import os; os.environ['OPENAI_API_KEY'] = 'here'
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)
```

!!! info "A few different LMs"

    === "OpenAI"
        You can authenticate by setting the `OPENAI_API_KEY` env variable or passing `api_key` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('openai/gpt-4o-mini', api_key='YOUR_OPENAI_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Gemini (AI Studio)"
        You can authenticate by setting the GEMINI_API_KEY env variable or passing `api_key` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('gemini/gemini-2.5-pro-preview-03-25', api_key='GEMINI_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Anthropic"
        You can authenticate by setting the ANTHROPIC_API_KEY env variable or passing `api_key` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('anthropic/claude-3-opus-20240229', api_key='YOUR_ANTHROPIC_API_KEY')
        dspy.configure(lm=lm)
        ```

    === "Databricks"
        If you're on the Databricks platform, authentication is automatic via their SDK. If not, you can set the env variables `DATABRICKS_API_KEY` and `DATABRICKS_API_BASE`, or pass `api_key` and `api_base` below.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('databricks/databricks-meta-llama-3-1-70b-instruct')
        dspy.configure(lm=lm)
        ```

    === "Local LMs on a GPU server"
          First, install [SGLang](https://sgl-project.github.io/start/install.html) and launch its server with your LM.

          ```bash
          > pip install "sglang[all]"
          > pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ 

          > CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path meta-llama/Meta-Llama-3-8B-Instruct
          ```

          Then, connect to it from your DSPy code as an OpenAI-compatible endpoint.

          ```python linenums="1"
          lm = dspy.LM("openai/meta-llama/Meta-Llama-3-8B-Instruct",
                           api_base="http://localhost:7501/v1",  # ensure this points to your port
                           api_key="", model_type='chat')
          dspy.configure(lm=lm)
          ```

    === "Local LMs on your laptop"
          First, install [Ollama](https://github.com/ollama/ollama) and launch its server with your LM.

          ```bash
          > curl -fsSL https://ollama.ai/install.sh | sh
          > ollama run llama3.2:1b
          ```

          Then, connect to it from your DSPy code.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('ollama_chat/llama3.2', api_base='http://localhost:11434', api_key='')
        dspy.configure(lm=lm)
        ```

    === "Other providers"
        In DSPy, you can use any of the dozens of [LLM providers supported by LiteLLM](https://docs.litellm.ai/docs/providers). Simply follow their instructions for which `{PROVIDER}_API_KEY` to set and how to write pass the `{provider_name}/{model_name}` to the constructor. 

        Some examples:

        - `anyscale/mistralai/Mistral-7B-Instruct-v0.1`, with `ANYSCALE_API_KEY`
        - `together_ai/togethercomputer/llama-2-70b-chat`, with `TOGETHERAI_API_KEY`
        - `sagemaker/<your-endpoint-name>`, with `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION_NAME`
        - `azure/<your_deployment_name>`, with `AZURE_API_KEY`, `AZURE_API_BASE`, `AZURE_API_VERSION`, and the optional `AZURE_AD_TOKEN` and `AZURE_API_TYPE` as environment variables. If you are initiating external models without setting environment variables, use the following:
        `lm = dspy.LM('azure/<your_deployment_name>', api_key = 'AZURE_API_KEY' , api_base = 'AZURE_API_BASE', api_version = 'AZURE_API_VERSION')`


        
        If your provider offers an OpenAI-compatible endpoint, just add an `openai/` prefix to your full model name.

        ```python linenums="1"
        import dspy
        lm = dspy.LM('openai/your-model-name', api_key='PROVIDER_API_KEY', api_base='YOUR_PROVIDER_URL')
        dspy.configure(lm=lm)
        ```
If you run into errors, please refer to the [LiteLLM Docs](https://docs.litellm.ai/docs/providers) to verify if you are using the same variable names/following the right procedure.

## Calling the LM directly.

It's easy to call the `lm` you configured above directly. This gives you a unified API and lets you benefit from utilities like automatic caching.

```python linenums="1"       
lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']
``` 

## Using the LM with DSPy modules.

Idiomatic DSPy involves using _modules_, which we discuss in the next guide.

```python linenums="1" 
# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought('question -> answer')

# Run with the default LM configured with `dspy.configure` above.
response = qa(question="How many floors are in the castle David Gregory inherited?")
print(response.answer)
```
**Possible Output:**
```text
The castle David Gregory inherited has 7 floors.
```

## Using multiple LMs.

You can change the default LM globally with `dspy.configure` or change it inside a block of code with `dspy.context`.

!!! tip
    Using `dspy.configure` and `dspy.context` is thread-safe!


```python linenums="1" 
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))
response = qa(question="How many floors are in the castle David Gregory inherited?")
print('GPT-4o-mini:', response.answer)

with dspy.context(lm=dspy.LM('openai/gpt-3.5-turbo')):
    response = qa(question="How many floors are in the castle David Gregory inherited?")
    print('GPT-3.5-turbo:', response.answer)
```
**Possible Output:**
```text
GPT-4o: The number of floors in the castle David Gregory inherited cannot be determined with the information provided.
GPT-3.5-turbo: The castle David Gregory inherited has 7 floors.
```

## Configuring LM generation.

For any LM, you can configure any of the following attributes at initialization or in each subsequent call.

```python linenums="1" 
gpt_4o_mini = dspy.LM('openai/gpt-4o-mini', temperature=0.9, max_tokens=3000, stop=None, cache=False)
```

By default LMs in DSPy are cached. If you repeat the same call, you will get the same outputs. But you can turn off caching by setting `cache=False`.


## Inspecting output and usage metadata.

Every LM object maintains the history of its interactions, including inputs, outputs, token usage (and $$$ cost), and metadata.

```python linenums="1" 
len(lm.history)  # e.g., 3 calls to the LM

lm.history[-1].keys()  # access the last call to the LM, with all metadata
```

**Output:**
```text
dict_keys(['prompt', 'messages', 'kwargs', 'response', 'outputs', 'usage', 'cost'])
```

### Advanced: Building custom LMs and writing your own Adapters.

Though rarely needed, you can write custom LMs by inheriting from `dspy.BaseLM`. Another advanced layer in the DSPy ecosystem is that of _adapters_, which sit between DSPy signatures and LMs. A future version of this guide will discuss these advanced features, though you likely don't need them.



### learn/programming/modules.md

---
sidebar_position: 3
---

# Modules

A **DSPy module** is a building block for programs that use LMs.

- Each built-in module abstracts a **prompting technique** (like chain of thought or ReAct). Crucially, they are generalized to handle any signature.

- A DSPy module has **learnable parameters** (i.e., the little pieces comprising the prompt and the LM weights) and can be invoked (called) to process inputs and return outputs.

- Multiple modules can be composed into bigger modules (programs). DSPy modules are inspired directly by NN modules in PyTorch, but applied to LM programs.


## How do I use a built-in module, like `dspy.Predict` or `dspy.ChainOfThought`?

Let's start with the most fundamental module, `dspy.Predict`. Internally, all other DSPy modules are built using `dspy.Predict`. We'll assume you are already at least a little familiar with [DSPy signatures](/learn/programming/signatures), which are declarative specs for defining the behavior of any module we use in DSPy.

To use a module, we first **declare** it by giving it a signature. Then we **call** the module with the input arguments, and extract the output fields!

```python
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

# 1) Declare with a signature.
classify = dspy.Predict('sentence -> sentiment: bool')

# 2) Call with input argument(s). 
response = classify(sentence=sentence)

# 3) Access the output.
print(response.sentiment)
```
**Output:**
```text
True
```

When we declare a module, we can pass configuration keys to it.

Below, we'll pass `n=5` to request five completions. We can also pass `temperature` or `max_len`, etc.

Let's use `dspy.ChainOfThought`. In many cases, simply swapping `dspy.ChainOfThought` in place of `dspy.Predict` improves quality.

```python
question = "What's something great about the ColBERT retrieval model?"

# 1) Declare with a signature, and pass some config.
classify = dspy.ChainOfThought('question -> answer', n=5)

# 2) Call with input argument.
response = classify(question=question)

# 3) Access the outputs.
response.completions.answer
```
**Possible Output:**
```text
['One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.',
 'Its ability to efficiently retrieve relevant information from large document collections.',
 'One great thing about the ColBERT retrieval model is its superior performance compared to other models and its efficient use of pre-trained language models.',
 'One great thing about the ColBERT retrieval model is its superior efficiency and accuracy compared to other models.',
 'One great thing about the ColBERT retrieval model is its ability to incorporate user feedback and support complex queries.']
```

Let's discuss the output object here. The `dspy.ChainOfThought` module will generally inject a `reasoning` before the output field(s) of your signature.

Let's inspect the (first) reasoning and answer!

```python
print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.answer}")
```
**Possible Output:**
```text
Reasoning: We can consider the fact that ColBERT has shown to outperform other state-of-the-art retrieval models in terms of efficiency and effectiveness. It uses contextualized embeddings and performs document retrieval in a way that is both accurate and scalable.
Answer: One great thing about the ColBERT retrieval model is its superior efficiency and effectiveness compared to other models.
```

This is accessible whether we request one or many completions.

We can also access the different completions as a list of `Prediction`s or as several lists, one for each field.

```python
response.completions[3].reasoning == response.completions.reasoning[3]
```
**Output:**
```text
True
```


## What other DSPy modules are there? How can I use them?

The others are very similar. They mainly change the internal behavior with which your signature is implemented!

1. **`dspy.Predict`**: Basic predictor. Does not modify the signature. Handles the key forms of learning (i.e., storing the instructions and demonstrations and updates to the LM).

2. **`dspy.ChainOfThought`**: Teaches the LM to think step-by-step before committing to the signature's response.

3. **`dspy.ProgramOfThought`**: Teaches the LM to output code, whose execution results will dictate the response.

4. **`dspy.ReAct`**: An agent that can use tools to implement the given signature.

5. **`dspy.MultiChainComparison`**: Can compare multiple outputs from `ChainOfThought` to produce a final prediction.


We also have some function-style modules:

6. **`dspy.majority`**: Can do basic voting to return the most popular response from a set of predictions.


!!! info "A few examples of DSPy modules on simple tasks."
    Try the examples below after configuring your `lm`. Adjust the fields to explore what tasks your LM can do well out of the box.

    === "Math"

        ```python linenums="1"
        math = dspy.ChainOfThought("question -> answer: float")
        math(question="Two dice are tossed. What is the probability that the sum equals two?")
        ```
        
        **Possible Output:**
        ```text
        Prediction(
            reasoning='When two dice are tossed, each die has 6 faces, resulting in a total of 6 x 6 = 36 possible outcomes. The sum of the numbers on the two dice equals two only when both dice show a 1. This is just one specific outcome: (1, 1). Therefore, there is only 1 favorable outcome. The probability of the sum being two is the number of favorable outcomes divided by the total number of possible outcomes, which is 1/36.',
            answer=0.0277776
        )
        ```

    === "Retrieval-Augmented Generation"

        ```python linenums="1"       
        def search(query: str) -> list[str]:
            """Retrieves abstracts from Wikipedia."""
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]
        
        rag = dspy.ChainOfThought('context, question -> response')

        question = "What's the name of the castle that David Gregory inherited?"
        rag(context=search(question), question=question)
        ```
        
        **Possible Output:**
        ```text
        Prediction(
            reasoning='The context provides information about David Gregory, a Scottish physician and inventor. It specifically mentions that he inherited Kinnairdy Castle in 1664. This detail directly answers the question about the name of the castle that David Gregory inherited.',
            response='Kinnairdy Castle'
        )
        ```

    === "Classification"

        ```python linenums="1"
        from typing import Literal

        class Classify(dspy.Signature):
            """Classify sentiment of a given sentence."""
            
            sentence: str = dspy.InputField()
            sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
            confidence: float = dspy.OutputField()

        classify = dspy.Predict(Classify)
        classify(sentence="This book was super fun to read, though not the last chapter.")
        ```
        
        **Possible Output:**

        ```text
        Prediction(
            sentiment='positive',
            confidence=0.75
        )
        ```

    === "Information Extraction"

        ```python linenums="1"        
        text = "Apple Inc. announced its latest iPhone 14 today. The CEO, Tim Cook, highlighted its new features in a press release."

        module = dspy.Predict("text -> title, headings: list[str], entities_and_metadata: list[dict[str, str]]")
        response = module(text=text)

        print(response.title)
        print(response.headings)
        print(response.entities_and_metadata)
        ```
        
        **Possible Output:**
        ```text
        Apple Unveils iPhone 14
        ['Introduction', 'Key Features', "CEO's Statement"]
        [{'entity': 'Apple Inc.', 'type': 'Organization'}, {'entity': 'iPhone 14', 'type': 'Product'}, {'entity': 'Tim Cook', 'type': 'Person'}]
        ```

    === "Agents"

        ```python linenums="1"       
        def evaluate_math(expression: str) -> float:
            return dspy.PythonInterpreter({}).execute(expression)

        def search_wikipedia(query: str) -> str:
            results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
            return [x['text'] for x in results]

        react = dspy.ReAct("question -> answer: float", tools=[evaluate_math, search_wikipedia])

        pred = react(question="What is 9362158 divided by the year of birth of David Gregory of Kinnairdy castle?")
        print(pred.answer)
        ```
        
        **Possible Output:**

        ```text
        5761.328
        ```


## How do I compose multiple modules into a bigger program?

DSPy is just Python code that uses modules in any control flow you like, with a little magic internally at `compile` time to trace your LM calls. What this means is that, you can just call the modules freely.

See tutorials like [multi-hop search](https://dspy.ai/tutorials/multihop_search/), whose module is reproduced below as an example.

```python linenums="1"        
class Hop(dspy.Module):
    def __init__(self, num_docs=10, num_hops=4):
        self.num_docs, self.num_hops = num_docs, num_hops
        self.generate_query = dspy.ChainOfThought('claim, notes -> query')
        self.append_notes = dspy.ChainOfThought('claim, notes, context -> new_notes: list[str], titles: list[str]')

    def forward(self, claim: str) -> list[str]:
        notes = []
        titles = []

        for _ in range(self.num_hops):
            query = self.generate_query(claim=claim, notes=notes).query
            context = search(query, k=self.num_docs)
            prediction = self.append_notes(claim=claim, notes=notes, context=context)
            notes.extend(prediction.new_notes)
            titles.extend(prediction.titles)
        
        return dspy.Prediction(notes=notes, titles=list(set(titles)))
```

## How do I track LM usage?

!!! note "Version Requirement"
    LM usage tracking is available in DSPy version 2.6.16 and later.

DSPy provides built-in tracking of language model usage across all module calls. To enable tracking:

```python
dspy.settings.configure(track_usage=True)
```

Once enabled, you can access usage statistics from any `dspy.Prediction` object:

```python
usage = prediction_instance.get_lm_usage()
```

The usage data is returned as a dictionary that maps each language model name to its usage statistics. Here's a complete example:

```python
import dspy

# Configure DSPy with tracking enabled
dspy.settings.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    track_usage=True
)

# Define a simple program that makes multiple LM calls
class MyProgram(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question -> answer")
        self.predict2 = dspy.ChainOfThought("question, answer -> score")

    def __call__(self, question: str) -> str:
        answer = self.predict1(question=question)
        score = self.predict2(question=question, answer=answer)
        return score

# Run the program and check usage
program = MyProgram()
output = program(question="What is the capital of France?")
print(output.get_lm_usage())
```

This will output usage statistics like:

```python
{
    'openai/gpt-4o-mini': {
        'completion_tokens': 61,
        'prompt_tokens': 260,
        'total_tokens': 321,
        'completion_tokens_details': {
            'accepted_prediction_tokens': 0,
            'audio_tokens': 0,
            'reasoning_tokens': 0,
            'rejected_prediction_tokens': 0,
            'text_tokens': None
        },
        'prompt_tokens_details': {
            'audio_tokens': 0,
            'cached_tokens': 0,
            'text_tokens': None,
            'image_tokens': None
        }
    }
}
```

When using DSPy's caching features (either in-memory or on-disk via litellm), cached responses won't count toward usage statistics. For example:

```python
# Enable caching
dspy.settings.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=True),
    track_usage=True
)

program = MyProgram()

# First call - will show usage statistics
output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Shows token usage

# Second call - same question, will use cache
output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Shows empty dict: {}
```


### learn/programming/overview.md

---
sidebar_position: 1
---

# Programming in DSPy

DSPy is a bet on _writing code instead of strings_. In other words, building the right control flow is crucial. Start by **defining your task**. What are the inputs to your system and what should your system produce as output? Is it a chatbot over your data or perhaps a code assistant? Or maybe a system for translation, for highlighting snippets from search results, or for generating reports with citations?

Next, **define your initial pipeline**. Can your DSPy program just be a single module or do you need to break it down into a few steps? Do you need retrieval or other tools, like a calculator or a calendar API? Is there a typical workflow for solving your problem in multiple well-scoped steps, or do you want more open-ended tool use with agents for your task? Think about these but start simple, perhaps with just a single `dspy.ChainOfThought` module, then add complexity incrementally based on observations.

As you do this, **craft and try a handful of examples** of the inputs to your program. Consider using a powerful LM at this point, or a couple of different LMs, just to understand what's possible. Record interesting (both easy and hard) examples you try. This will be useful when you are doing evaluation and optimization later.


??? "Beyond encouraging good design patterns, how does DSPy help here?"

    Conventional prompts couple your fundamental system architecture with incidental choices not portable to new LMs, objectives, or pipelines. A conventional prompt asks the LM to take some inputs and produce some outputs of certain types (a _signature_), formats the inputs in certain ways and requests outputs in a form it can parse accurately (an _adapter_), asks the LM to apply certain strategies like "thinking step by step" or using tools (a _module_'s logic), and relies on substantial trial-and-error to discover the right way to ask each LM to do this (a form of manual _optimization_).
    
    DSPy separates these concerns and automates the lower-level ones until you need to consider them. This allow you to write much shorter code, with much higher portability. For example, if you write a program using DSPy modules, you can swap the LM or its adapter without changing the rest of your logic. Or you can exchange one _module_, like `dspy.ChainOfThought`, with another, like `dspy.ProgramOfThought`, without modifying your signatures. When you're ready to use optimizers, the same program can have its prompts optimized or its LM weights fine-tuned.


### learn/programming/signatures.md

---
sidebar_position: 2
---

# Signatures

When we assign tasks to LMs in DSPy, we specify the behavior we need as a Signature.

**A signature is a declarative specification of input/output behavior of a DSPy module.** Signatures allow you to tell the LM _what_ it needs to do, rather than specify _how_ we should ask the LM to do it.

You're probably familiar with function signatures, which specify the input and output arguments and their types. DSPy signatures are similar, but with a couple of differences. While typical function signatures just _describe_ things, DSPy Signatures _declare and initialize the behavior_ of modules. Moreover, the field names matter in DSPy Signatures. You express semantic roles in plain English: a `question` is different from an `answer`, a `sql_query` is different from `python_code`.

## Why should I use a DSPy Signature?

For modular and clean code, in which LM calls can be optimized into high-quality prompts (or automatic finetunes). Most people coerce LMs to do tasks by hacking long, brittle prompts. Or by collecting/generating data for fine-tuning. Writing signatures is far more modular, adaptive, and reproducible than hacking at prompts or finetunes. The DSPy compiler will figure out how to build a highly-optimized prompt for your LM (or finetune your small LM) for your signature, on your data, and within your pipeline. In many cases, we found that compiling leads to better prompts than humans write. Not because DSPy optimizers are more creative than humans, but simply because they can try more things and tune the metrics directly.

## **Inline** DSPy Signatures

Signatures can be defined as a short string, with argument names and optional types that define semantic roles for inputs/outputs.

1. Question Answering: `"question -> answer"`, which is equivalent to `"question: str -> answer: str"` as the default type is always `str`

2. Sentiment Classification: `"sentence -> sentiment: bool"`, e.g. `True` if positive

3. Summarization: `"document -> summary"`

Your signatures can also have multiple input/output fields with types:

4. Retrieval-Augmented Question Answering: `"context: list[str], question: str -> answer: str"`

5. Multiple-Choice Question Answering with Reasoning: `"question, choices: list[str] -> reasoning: str, selection: int"`

**Tip:** For fields, any valid variable names work! Field names should be semantically meaningful, but start simple and don't prematurely optimize keywords! Leave that kind of hacking to the DSPy compiler. For example, for summarization, it's probably fine to say `"document -> summary"`, `"text -> gist"`, or `"long_context -> tldr"`.

You can also add instructions to your inline signature, which can use variables at runtime. Use the `instructions` keyword argument to add instructions to your signature.

```python
toxicity = dspy.Predict(
    dspy.Signature(
        "comment -> toxic: bool",
        instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks.",
    )
)
```

### Example A: Sentiment Classification

```python
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

classify = dspy.Predict('sentence -> sentiment: bool')  # we'll see an example with Literal[] later
classify(sentence=sentence).sentiment
```
**Output:**
```text
True
```

### Example B: Summarization

```python
# Example from the XSum dataset.
document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = dspy.ChainOfThought('document -> summary')
response = summarize(document=document)

print(response.summary)
```
**Possible Output:**
```text
The 21-year-old Lee made seven appearances and scored one goal for West Ham last season. He had loan spells in League One with Blackpool and Colchester United, scoring twice for the latter. He has now signed a contract with Barnsley, but the length of the contract has not been revealed.
```

Many DSPy modules (except `dspy.Predict`) return auxiliary information by expanding your signature under the hood.

For example, `dspy.ChainOfThought` also adds a `reasoning` field that includes the LM's reasoning before it generates the output `summary`.

```python
print("Reasoning:", response.reasoning)
```
**Possible Output:**
```text
Reasoning: We need to highlight Lee's performance for West Ham, his loan spells in League One, and his new contract with Barnsley. We also need to mention that his contract length has not been disclosed.
```

## **Class-based** DSPy Signatures

For some advanced tasks, you need more verbose signatures. This is typically to:

1. Clarify something about the nature of the task (expressed below as a `docstring`).

2. Supply hints on the nature of an input field, expressed as a `desc` keyword argument for `dspy.InputField`.

3. Supply constraints on an output field, expressed as a `desc` keyword argument for `dspy.OutputField`.

### Example C: Classification

```python
from typing import Literal

class Emotion(dspy.Signature):
    """Classify emotion."""
    
    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField()

sentence = "i started feeling a little vulnerable when the giant spotlight started blinding me"  # from dair-ai/emotion

classify = dspy.Predict(Emotion)
classify(sentence=sentence)
```
**Possible Output:**
```text
Prediction(
    sentiment='fear'
)
```

**Tip:** There's nothing wrong with specifying your requests to the LM more clearly. Class-based Signatures help you with that. However, don't prematurely tune the keywords of your signature by hand. The DSPy optimizers will likely do a better job (and will transfer better across LMs).

### Example D: A metric that evaluates faithfulness to citations

```python
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""

    context: str = dspy.InputField(desc="facts here are assumed to be true")
    text: str = dspy.InputField()
    faithfulness: bool = dspy.OutputField()
    evidence: dict[str, list[str]] = dspy.OutputField(desc="Supporting evidence for claims")

context = "The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."

text = "Lee scored 3 goals for Colchester United."

faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
faithfulness(context=context, text=text)
```
**Possible Output:**
```text
Prediction(
    reasoning="Let's check the claims against the context. The text states Lee scored 3 goals for Colchester United, but the context clearly states 'He scored twice for the U's'. This is a direct contradiction.",
    faithfulness=False,
    evidence={'goal_count': ["scored twice for the U's"]}
)
```

### Example E: Multi-modal image classification

```python
class DogPictureSignature(dspy.Signature):
    """Output the dog breed of the dog in the image."""
    image_1: dspy.Image = dspy.InputField(desc="An image of a dog")
    answer: str = dspy.OutputField(desc="The dog breed of the dog in the image")

image_url = "https://picsum.photos/id/237/200/300"
classify = dspy.Predict(DogPictureSignature)
classify(image_1=dspy.Image.from_url(image_url))
```

**Possible Output:**

```text
Prediction(
    answer='Labrador Retriever'
)
```

## Type Resolution in Signatures

DSPy signatures support various annotation types:

1. **Basic types** like `str`, `int`, `bool`
2. **Typing module types** like `List[str]`, `Dict[str, int]`, `Optional[float]`. `Union[str, int]`
3. **Custom types** defined in your code
4. **Dot notation** for nested types with proper configuration
5. **Special data types** like `dspy.Image, dspy.History`

### Working with Custom Types

```python
# Simple custom type
class QueryResult(pydantic.BaseModel):
    text: str
    score: float

signature = dspy.Signature("query: str -> result: QueryResult")

class Container:
    class Query(pydantic.BaseModel):
        text: str
    class Score(pydantic.BaseModel):
        score: float

signature = dspy.Signature("query: Container.Query -> score: Container.Score")
```

## Using signatures to build modules & compiling them

While signatures are convenient for prototyping with structured inputs/outputs, that's not the only reason to use them!

You should compose multiple signatures into bigger [DSPy modules](modules.md) and [compile these modules into optimized prompts](../optimization/optimizers.md) and finetunes.


## Tutorials

### tutorials/agents/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/agents/index.ipynb, nbconvert not installed.

### tutorials/async/index.md

# Async DSPy Programming

DSPy provides native support for asynchronous programming, allowing you to build more efficient and
scalable applications. This guide will walk you through how to leverage async capabilities in DSPy,
covering both built-in modules and custom implementations.

## Why Use Async in DSPy?

Asynchronous programming in DSPy offers several benefits:
- Improved performance through concurrent operations
- Better resource utilization
- Reduced waiting time for I/O-bound operations
- Enhanced scalability for handling multiple requests

## When Should I use Sync or Async?

Choosing between synchronous and asynchronous programming in DSPy depends on your specific use case.
Here's a guide to help you make the right choice:

Use Synchronous Programming When

- You're exploring or prototyping new ideas
- You're conducting research or experiments
- You're building small to medium-sized applications
- You need simpler, more straightforward code
- You want easier debugging and error tracking

Use Asynchronous Programming When:

- You're building a high-throughput service (high QPS)
- You're working with tools that only support async operations
- You need to handle multiple concurrent requests efficiently
- You're building a production service that requires high scalability

### Important Considerations

While async programming offers performance benefits, it comes with some trade-offs:

- More complex error handling and debugging
- Potential for subtle, hard-to-track bugs
- More complex code structure
- Different code between ipython (Colab, Jupyter lab, Databricks notebooks, ...) and normal python runtime.

We recommend starting with synchronous programming for most development scenarios and switching to async
only when you have a clear need for its benefits. This approach allows you to focus on the core logic of
your application before dealing with the additional complexity of async programming.

## Using Built-in Modules Asynchronously

Most DSPy built-in modules support asynchronous operations through the `acall()` method. This method
maintains the same interface as the synchronous `__call__` method but operates asynchronously.

Here's a basic example using `dspy.Predict`:

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
predict = dspy.Predict("question->answer")

async def main():
    # Use acall() for async execution
    output = await predict.acall(question="why did a chicken cross the kitchen?")
    print(output)


asyncio.run(main())
```

### Working with Async Tools

DSPy's `Tool` class seamlessly integrates with async functions. When you provide an async
function to `dspy.Tool`, you can execute it using `acall()`. This is particularly useful
for I/O-bound operations or when working with external services.

```python
import asyncio
import dspy
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"

async def foo(x):
    # Simulate an async operation
    await asyncio.sleep(0.1)
    print(f"I get: {x}")

# Create a tool from the async function
tool = dspy.Tool(foo)

async def main():
    # Execute the tool asynchronously
    await tool.acall(x=2)

asyncio.run(main())
```

Note: When using `dspy.ReAct` with tools, calling `acall()` on the ReAct instance will automatically
execute all tools asynchronously using their `acall()` methods.

## Creating Custom Async DSPy Modules

To create your own async DSPy module, implement the `aforward()` method instead of `forward()`. This method
should contain your module's async logic. Here's an example of a custom module that chains two async operations:

```python
import dspy
import asyncio
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class MyModule(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question->answer")
        self.predict2 = dspy.ChainOfThought("answer->simplified_answer")

    async def aforward(self, question, **kwargs):
        # Execute predictions sequentially but asynchronously
        answer = await self.predict1.acall(question=question)
        return await self.predict2.acall(answer=answer)


async def main():
    mod = MyModule()
    result = await mod.acall(question="Why did a chicken cross the kitchen?")
    print(result)


asyncio.run(main())
```


### tutorials/audio/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/audio/index.ipynb, nbconvert not installed.

### tutorials/build_ai_program/index.md

- [Build AI Agents with DSPy](/tutorials/customer_service_agent/)
- [Retrieval-Augmented Generation (RAG)](/tutorials/rag/)
- [Building RAG as Agent](/tutorials/agents/)
- [Entity Extraction](/tutorials/entity_extraction/)
- [Classification](/tutorials/classification/)
- [Multi-Hop RAG](/tutorials/multihop_search/)
- [Privacy-Conscious Delegation](/tutorials/papillon/)
- [Program Of Thought](/tutorials/program_of_thought/)
- [Image Generation Prompt iteration](/tutorials/image_generation_prompting/)
- [Audio](/tutorials/audio/)


### tutorials/cache/index.md

# Use and Customize DSPy Cache

In this tutorial, we will explore the design of DSPy's caching mechanism and demonstrate how to effectively use and customize it.

## DSPy Cache Structure

DSPy's caching system is architected in three distinct layers:

1.  **In-memory cache**: Implemented using `cachetools.LRUCache`, this layer provides fast access to frequently used data.
2.  **On-disk cache**: Leveraging `diskcache.FanoutCache`, this layer offers persistent storage for cached items.
3.  **Prompt cache (Server-side cache)**: This layer is managed by the LLM service provider (e.g., OpenAI, Anthropic).

While DSPy does not directly control the server-side prompt cache, it offers users the flexibility to enable, disable, and customize the in-memory and on-disk caches to suit their specific requirements.

## Using DSPy Cache

By default, both in-memory and on-disk caching are automatically enabled in DSPy. No specific action is required to start using the cache. When a cache hit occurs, you will observe a significant reduction in the module call's execution time. Furthermore, if usage tracking is enabled, the usage metrics for a cached call will be `None`.

Consider the following example:

```python
import dspy
import os
import time

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"), track_usage=True)

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of basketball?")
print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result1.get_lm_usage()}")

start = time.time()
result2 = predict(question="Who is the GOAT of basketball?")
print(f"Time elapse: {time.time() - start: 2f}\n\nTotal usage: {result2.get_lm_usage()}")
```

A sample output looks like:

```
Time elapse:  4.384113
Total usage: {'openai/gpt-4o-mini': {'completion_tokens': 97, 'prompt_tokens': 144, 'total_tokens': 241, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0, 'text_tokens': None, 'image_tokens': None}}}

Time elapse:  0.000529
Total usage: {}
```

## Disabling/Enabling DSPy Cache

There are scenarios where you might need to disable caching, either entirely or selectively for in-memory or on-disk caches. For instance:

- You require different responses for identical LM requests.
- You lack disk write permissions and need to disable the on-disk cache.
- You have limited memory resources and wish to disable the in-memory cache.

DSPy provides the `dspy.configure_cache()` utility function for this purpose. You can use the corresponding flags to control the enabled/disabled state of each cache type:

```python
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)
```

In additions, you can manage the capacity of the in-memory and on-disk caches:

```python
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_size_limit_bytes=YOUR_DESIRED_VALUE,
    memory_max_entries=YOUR_DESIRED_VALUE,
)
```

Please note that `disk_size_limit_bytes` defines the maximum size in bytes for the on-disk cache, while `memory_max_entries` specifies the maximum number of entries for the in-memory cache.

## Understanding and Customizing the Cache

In specific situations, you might want to implement a custom cache, for example, to gain finer control over how cache keys are generated. By default, the cache key is derived from a hash of all request arguments sent to `litellm`, excluding credentials like `api_key`.

To create a custom cache, you need to subclass `dspy.clients.Cache` and override the relevant methods:

```python
class CustomCache(dspy.clients.Cache):
    def __init__(self, **kwargs):
        {write your own constructor}

    def cache_key(self, request: Dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        {write your logic of computing cache key}

    def get(self, request: Dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> Any:
        {write your cache read logic}

    def put(
        self,
        request: Dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: Optional[list[str]] = None,
        enable_memory_cache: bool = True,
    ) -> None:
        {write your cache write logic}
```

To ensure seamless integration with the rest of DSPy, it is recommended to implement your custom cache using the same method signatures as the base class, or at a minimum, include `**kwargs` in your method definitions to prevent runtime errors during cache read/write operations.

Once your custom cache class is defined, you can instruct DSPy to use it:

```python
dspy.cache = CustomCache()
```

Let's illustrate this with a practical example. Suppose we want the cache key computation to depend solely on the request message content, ignoring other parameters like the specific LM being called. We can create a custom cache as follows:

```python
class CustomCache(dspy.clients.Cache):

    def cache_key(self, request: Dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        messages = request.get("messages", [])
        return sha256(ujson.dumps(messages, sort_keys=True).encode()).hexdigest()

dspy.cache = CustomCache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=dspy.clients.DISK_CACHE_DIR)
```

For comparison, consider executing the code below without the custom cache:

```python
import dspy
import os
import time

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of soccer?")
print(f"Time elapse: {time.time() - start: 2f}")

start = time.time()
with dspy.context(lm=dspy.LM("openai/gpt-4.1-mini")):
    result2 = predict(question="Who is the GOAT of soccer?")
print(f"Time elapse: {time.time() - start: 2f}")
```

The time elapsed will indicate that the cache is not hit on the second call. However, when using the custom cache:

```python
import dspy
import os
import time
from typing import Dict, Any, Optional
import ujson
from hashlib import sha256

os.environ["OPENAI_API_KEY"] = "{your_openai_key}"

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

class CustomCache(dspy.clients.Cache):

    def cache_key(self, request: Dict[str, Any], ignored_args_for_cache_key: Optional[list[str]] = None) -> str:
        messages = request.get("messages", [])
        return sha256(ujson.dumps(messages, sort_keys=True).encode()).hexdigest()

dspy.cache = CustomCache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=dspy.clients.DISK_CACHE_DIR)

predict = dspy.Predict("question->answer")

start = time.time()
result1 = predict(question="Who is the GOAT of volleyball?")
print(f"Time elapse: {time.time() - start: 2f}")

start = time.time()
with dspy.context(lm=dspy.LM("openai/gpt-4.1-mini")):
    result2 = predict(question="Who is the GOAT of volleyball?")
print(f"Time elapse: {time.time() - start: 2f}")
```

You will observe that the cache is hit on the second call, demonstrating the effect of the custom cache key logic.

### tutorials/classification/index.md

Please refer to [this tutorial from Drew Breunig](https://www.dbreunig.com/2024/12/12/pipelines-prompt-optimization-with-dspy.html) using DSPy.

This tutorial demonstrates a few aspects of using DSPy in a highly-accessible, concrete context for categorizing historic events with a tiny LM.

### tutorials/classification_finetuning/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/classification_finetuning/index.ipynb, nbconvert not installed.

### tutorials/core_development/index.md

- [Use MCP in DSPy](/tutorials/mcp/)
- [Output Refinement](/tutorials/output_refinement/best-of-n-and-refine/)
- [Saving and Loading](/tutorials/saving/)
- [Cache](/tutorials/cache/)
- [Deployment](/tutorials/deployment/)
- [Debugging & Observability](/tutorials/observability/)
- [Tracking DSPy Optimizers](/tutorials/optimizer_tracking/)
- [Streaming](/tutorials/streaming/)
- [Async](/tutorials/async/)

### tutorials/customer_service_agent/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/customer_service_agent/index.ipynb, nbconvert not installed.

### tutorials/deployment/index.md

# Tutorial: Deploying your DSPy program

This guide demonstrates two potential ways to deploy your DSPy program in production: FastAPI for lightweight deployments and MLflow for more production-grade deployments with program versioning and management.

Below, we'll assume you have the following simple DSPy program that you want to deploy. You can replace this with something more sophisticated.

```python
import dspy

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))
dspy_program = dspy.ChainOfThought("question -> answer")
```

## Deploying with FastAPI

FastAPI offers a straightforward way to serve your DSPy program as a REST API. This is ideal when you have direct access to your program code and need a lightweight deployment solution.

```bash
> pip install fastapi uvicorn
> export OPENAI_API_KEY="your-openai-api-key"
```

Let's create a FastAPI application to serve your `dspy_program` defined above.

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import dspy

app = FastAPI(
    title="DSPy Program API",
    description="A simple API serving a DSPy Chain of Thought program",
    version="1.0.0"
)

# Define request model for better documentation and validation
class Question(BaseModel):
    text: str

# Configure your language model and 'asyncify' your DSPy program.
lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm, async_max_workers=4) # default is 8
dspy_program = dspy.ChainOfThought("question -> answer")
dspy_program = dspy.asyncify(dspy_program)

@app.post("/predict")
async def predict(question: Question):
    try:
        result = await dspy_program(question=question.text)
        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

In the code above, we call `dspy.asyncify` to convert the dspy program to run in async mode for high-throughput FastAPI
deployments. Currently, this runs the dspy program in a separate thread and awaits its result.

By default, the limit of spawned threads is 8. Think of this like a worker pool.
If you have 8 in-flight programs and call it once more, the 9th call will wait until one of the 8 returns.
You can configure the async capacity using the new `async_max_workers` setting.

??? "Streaming, in DSPy 2.6.0+"

    Streaming is also supported in DSPy 2.6.0+, available as a release candidate via `pip install -U --pre dspy`.

    We can use `dspy.streamify` to convert the dspy program to a streaming mode. This is useful when you want to stream
    the intermediate outputs (i.e. O1-style reasoning) to the client before the final prediction is ready. This uses
    asyncify under the hood and inherits the execution semantics.

    ```python
    dspy_program = dspy.asyncify(dspy.ChainOfThought("question -> answer"))
    streaming_dspy_program = dspy.streamify(dspy_program)

    @app.post("/predict/stream")
    async def stream(question: Question):
        async def generate():
            async for value in streaming_dspy_program(question=question.text):
                if isinstance(value, dspy.Prediction):
                    data = {"prediction": value.labels().toDict()}
                elif isinstance(value, litellm.ModelResponse):
                    data = {"chunk": value.json()}
                yield f"data: {ujson.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Since you're often going to want to stream the result of a DSPy program as server-sent events,
    # we've included a helper function for that, which is equivalent to the code above.

    from dspy.utils.streaming import streaming_response

    @app.post("/predict/stream")
    async def stream(question: Question):
        stream = streaming_dspy_program(question=question.text)
        return StreamingResponse(streaming_response(stream), media_type="text/event-stream")
    ```

Write your code to a file, e.g., `fastapi_dspy.py`. Then you can serve the app with:

```bash
> uvicorn fastapi_dspy:app --reload
```

It will start a local server at `http://127.0.0.1:8000/`. You can test it with the python code below:

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"text": "What is the capital of France?"}
)
print(response.json())
```

You should see the response like below:

```json
{
  "status": "success",
  "data": {
    "reasoning": "The capital of France is a well-known fact, commonly taught in geography classes and referenced in various contexts. Paris is recognized globally as the capital city, serving as the political, cultural, and economic center of the country.",
    "answer": "The capital of France is Paris."
  }
}
```

## Deploying with MLflow

We recommend deploying with MLflow if you are looking to package your DSPy program and deploy in an isolated environment.
MLflow is a popular platform for managing machine learning workflows, including versioning, tracking, and deployment.

```bash
> pip install mlflow>=2.18.0
```

Let's spin up the MLflow tracking server, where we will store our DSPy program. The command below will start a local server at
`http://127.0.0.1:5000/`.

```bash
> mlflow ui
```

Then we can define the DSPy program and log it to the MLflow server. "log" is an overloaded term in MLflow, basically it means
we store the program information along with environment requirements in the MLflow server. This is done via the `mlflow.dspy.log_model()`
function, please see the code below:

> [!NOTE]
> As of MLflow 2.22.0, there is a caveat that you must wrap your DSPy program in a custom DSPy Module class when deploying with MLflow.
> This is because MLflow requires positional arguments while DSPy pre-built modules disallow positional arguments, e.g., `dspy.Predict`
> or `dspy.ChainOfThought`. To work around this, create a wrapper class that inherits from `dspy.Module` and implement your program's
> logic in the `forward()` method, as shown in the example below.

```python
import dspy
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("deploy_dspy_program")

lm = dspy.LM("openai/gpt-4o-mini")
dspy.settings.configure(lm=lm)

class MyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> answer")

    def forward(self, messages):
        return self.cot(question=messages[0]["content"])

dspy_program = MyProgram()

with mlflow.start_run():
    mlflow.dspy.log_model(
        dspy_program,
        "dspy_program",
        input_example={"messages": [{"role": "user", "content": "What is LLM agent?"}]},
        task="llm/v1/chat",
    )
```

We recommend you to set `task="llm/v1/chat"` so that the deployed program automatically takes input and generate output in
the same format as the OpenAI chat API, which is a common interface for LM applications. Write the code above into
a file, e.g. `mlflow_dspy.py`, and run it.

After you logged the program, you can view the saved information in MLflow UI. Open `http://127.0.0.1:5000/` and select
the `deploy_dspy_program` experiment, then select the run your just created, under the `Artifacts` tab, you should see the
logged program information, similar to the following screenshot:

![MLflow UI](./dspy_mlflow_ui.png)

Grab your run id from UI (or the console print when you execute `mlflow_dspy.py`), now you can deploy the logged program
with the following command:

```bash
> mlflow models serve -m runs:/{run_id}/model -p 6000
```

After the program is deployed, you can test it with the following command:

```bash
> curl http://127.0.0.1:6000/invocations -H "Content-Type:application/json"  --data '{"messages": [{"content": "what is 2 + 2?", "role": "user"}]}'
```

You should see the response like below:

```json
{
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"reasoning\": \"The question asks for the sum of 2 and 2. To find the answer, we simply add the two numbers together: 2 + 2 = 4.\", \"answer\": \"4\"}"
      },
      "finish_reason": "stop"
    }
  ]
}
```

For complete guide on how to deploy a DSPy program with MLflow, and how to customize the deployment, please refer to the
[MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).

### Best Practices for MLflow Deployment

1. **Environment Management**: Always specify your Python dependencies in a `conda.yaml` or `requirements.txt` file.
2. **Versioning**: Use meaningful tags and descriptions for your model versions.
3. **Input Validation**: Define clear input schemas and examples.
4. **Monitoring**: Set up proper logging and monitoring for production deployments.

For production deployments, consider using MLflow with containerization:

```bash
> mlflow models build-docker -m "runs:/{run_id}/model" -n "dspy-program"
> docker run -p 6000:8080 dspy-program
```

For a complete guide on production deployment options and best practices, refer to the
[MLflow documentation](https://mlflow.org/docs/latest/llms/dspy/index.html).


### tutorials/entity_extraction/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/entity_extraction/index.ipynb, nbconvert not installed.

### tutorials/games/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/games/index.ipynb, nbconvert not installed.

### tutorials/image_generation_prompting/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/image_generation_prompting/index.ipynb, nbconvert not installed.

### tutorials/index.md

Welcome to DSPy tutorials! We've organized our tutorials into three main categories to help you get started:

- **Build AI Programs with DSPy**: These hands-on tutorials guide you through building production-ready AI
  applications. From implementing RAG systems to creating intelligent agents, each tutorial demonstrates
  practical use cases. You'll also learn how to leverage DSPy optimizers to enhance your program's performance.

- **Optimize AI Programs with DSPy Optimizers**: These tutorials deep dive into DSPy's optimization capabilities. While
  lighter on programming concepts, they focus on how to systematically improve your AI programs using DSPy
  optimizers, and showcase how DSPy optimizers help improve the quality automatically.

- **DSPy Core Development**: These tutorials cover essential DSPy features and best practices. Learn how to implement
  key functionalities like streaming, caching, deployment, and monitoring in your DSPy applications.


- Build AI Programs with DSPy
    - [Retrieval-Augmented Generation (RAG)](/tutorials/rag/)
    - [Building RAG as Agent](/tutorials/agents/)
    - [Build AI Agents with DSPy](/tutorials/customer_service_agent/)
    - [Entity Extraction](/tutorials/entity_extraction/)
    - [Classification](/tutorials/classification/)
    - [Multi-Hop RAG](/tutorials/multihop_search/)
    - [Privacy-Conscious Delegation](/tutorials/papillon/)
    - [Program Of Thought](/tutorials/program_of_thought/)
    - [Image Generation Prompt iteration](/tutorials/image_generation_prompting/)
    - [Audio](/tutorials/audio/)


- Optimize AI Programs with DSPy
    - [Math Reasoning](/tutorials/math/)
    - [Classification Finetuning](/tutorials/classification_finetuning/)
    - [Advanced Tool Use](/tutorials/tool_use/)
    - [Finetuning Agents](/tutorials/games/)

- Tools, Development, and Deployment
    - [Use MCP in DSPy](/tutorials/mcp/)
    - [Output Refinement](/tutorials/output_refinement/best-of-n-and-refine/)
    - [Saving and Loading](/tutorials/saving/)
    - [Cache](/tutorials/cache/)
    - [Deployment](/tutorials/deployment/)
    - [Debugging & Observability](/tutorials/observability/)
    - [Tracking DSPy Optimizers](/tutorials/optimizer_tracking/)
    - [Streaming](/tutorials/streaming/)
    - [Async](/tutorials/async/)




### tutorials/math/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/math/index.ipynb, nbconvert not installed.

### tutorials/mcp/index.md

# Tutorial: Use MCP tools in DSPy

MCP, standing for Model Context Protocol, is an open protocol that standardizes how applications
provide context to LLMs. Despite some development overhead, MCP offers a valuable opportunity to
share tools, resources, and prompts with other developers regardless of the technical stack you are
using. Likewise, you can use the tools built by other developers without rewriting code.

In this guide, we will walk you through how to use MCP tools in DSPy. For demonstration purposes,
we will build an airline service agent that can help users book flights and modify or cancel
existing bookings. This will rely on an MCP server with custom tools, but it should be easy to generalize
to [MCP servers built by the community](https://modelcontextprotocol.io/examples).

??? "How to run this tutorial"
    This tutorial cannot be run in hosted IPython notebooks like Google Colab or Databricks notebooks.
    To run the code, you will need to follow the guide to write code on your local device. The code
    is tested on macOS and should work the same way in Linux environments.

## Install Dependencies

Before starting, let's install the required dependencies:

```shell
pip install -U dspy mcp
```

## MCP Server Setup

Let's first set up the MCP server for the airline agent, which contains:

- A set of databases
  - User database, storing user information.
  - Flight database, storing flight information.
  - Ticket database, storing customer tickets.
- A set of tools
  - fetch_flight_info: get flight information for specific dates.
  - fetch_itinerary: get information about booked itineraries.
  - book_itinerary: book a flight on behalf of the user.
  - modify_itinerary: modify an itinerary, either through flight changes or cancellation.
  - get_user_info: get user information.
  - file_ticket: file a backlog ticket for human assistance.

In your working directory, create a file `mcp_server.py`, and paste the following content into
it:

```python
import random
import string

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

# Create an MCP server
mcp = FastMCP("Airline Agent")


class Date(BaseModel):
    # Somehow LLM is bad at specifying `datetime.datetime`
    year: int
    month: int
    day: int
    hour: int


class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str


class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float


class Itinerary(BaseModel):
    confirmation_number: str
    user_profile: UserProfile
    flight: Flight


class Ticket(BaseModel):
    user_request: str
    user_profile: UserProfile


user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}

flight_database = {
    "DA123": Flight(
        flight_id="DA123",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=1),
        duration=3,
        price=200,
    ),
    "DA125": Flight(
        flight_id="DA125",
        origin="SFO",
        destination="JFK",
        date_time=Date(year=2025, month=9, day=1, hour=7),
        duration=9,
        price=500,
    ),
    "DA456": Flight(
        flight_id="DA456",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=1),
        duration=2,
        price=100,
    ),
    "DA460": Flight(
        flight_id="DA460",
        origin="SFO",
        destination="SNA",
        date_time=Date(year=2025, month=10, day=1, hour=9),
        duration=2,
        price=120,
    ),
}

itinery_database = {}
ticket_database = {}


@mcp.tool()
def fetch_flight_info(date: Date, origin: str, destination: str):
    """Fetch flight information from origin to destination on the given date"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ):
            flights.append(flight)
    return flights


@mcp.tool()
def fetch_itinerary(confirmation_number: str):
    """Fetch a booked itinerary information from database"""
    return itinery_database.get(confirmation_number)


@mcp.tool()
def pick_flight(flights: list[Flight]):
    """Pick up the best flight that matches users' request."""
    sorted_flights = sorted(
        flights,
        key=lambda x: (
            x.get("duration") if isinstance(x, dict) else x.duration,
            x.get("price") if isinstance(x, dict) else x.price,
        ),
    )
    return sorted_flights[0]


def generate_id(length=8):
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choices(chars, k=length))


@mcp.tool()
def book_itinerary(flight: Flight, user_profile: UserProfile):
    """Book a flight on behalf of the user."""
    confirmation_number = generate_id()
    while confirmation_number in itinery_database:
        confirmation_number = generate_id()
    itinery_database[confirmation_number] = Itinerary(
        confirmation_number=confirmation_number,
        user_profile=user_profile,
        flight=flight,
    )
    return confirmation_number, itinery_database[confirmation_number]


@mcp.tool()
def cancel_itinerary(confirmation_number: str, user_profile: UserProfile):
    """Cancel an itinerary on behalf of the user."""
    if confirmation_number in itinery_database:
        del itinery_database[confirmation_number]
        return
    raise ValueError("Cannot find the itinerary, please check your confirmation number.")


@mcp.tool()
def get_user_info(name: str):
    """Fetch the user profile from database with given name."""
    return user_database.get(name)


@mcp.tool()
def file_ticket(user_request: str, user_profile: UserProfile):
    """File a customer support ticket if this is something the agent cannot handle."""
    ticket_id = generate_id(length=6)
    ticket_database[ticket_id] = Ticket(
        user_request=user_request,
        user_profile=user_profile,
    )
    return ticket_id


if __name__ == "__main__":
    mcp.run()
```

Before we start the server, let's take a look at the code.

We first create a `FastMCP` instance, which is a utility that helps quickly build an MCP server:

```python
mcp = FastMCP("Airline Agent")
```

Then we define our data structures, which in a real-world application would be the database schema, e.g.:

```python
class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float
```

Following that, we initialize our database instances. In a real-world application, these would be connectors to
actual databases, but for simplicity, we just use dictionaries:

```python
user_database = {
    "Adam": UserProfile(user_id="1", name="Adam", email="adam@gmail.com"),
    "Bob": UserProfile(user_id="2", name="Bob", email="bob@gmail.com"),
    "Chelsie": UserProfile(user_id="3", name="Chelsie", email="chelsie@gmail.com"),
    "David": UserProfile(user_id="4", name="David", email="david@gmail.com"),
}
```

The next step is to define the tools and mark them with `@mcp.tool()` so that they are discoverable by
MCP clients as MCP tools:

```python
@mcp.tool()
def fetch_flight_info(date: Date, origin: str, destination: str):
    """Fetch flight information from origin to destination on the given date"""
    flights = []

    for flight_id, flight in flight_database.items():
        if (
            flight.date_time.year == date.year
            and flight.date_time.month == date.month
            and flight.date_time.day == date.day
            and flight.origin == origin
            and flight.destination == destination
        ):
            flights.append(flight)
    return flights
```

The last step is spinning up the server:

```python
if __name__ == "__main__":
    mcp.run()
```

Now we have finished writing the server! Let's launch it:

```shell
python path_to_your_working_directory/mcp_server.py
```

## Write a DSPy Program That Utilizes Tools in MCP Server

Now that the server is running, let's build the actual airline service agent which
utilizes the MCP tools in our server to assist users. In your working directory,
create a file named `dspy_mcp_agent.py`, and follow the guide to add code to it.

### Gather Tools from MCP Servers

We first need to gather all available tools from the MCP server and make them
usable by DSPy. DSPy provides an API [`dspy.Tool`](https://dspy.ai/api/primitives/Tool/)
as the standard tool interface. Let's convert all the MCP tools to `dspy.Tool`.

We need to create an MCP client instance to communicate with the MCP server, fetch all available
tools, and convert them to `dspy.Tool` using the static method `from_mcp_tool`:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["path_to_your_working_directory/mcp_server.py"],
    env=None,
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            print(len(dspy_tools))
            print(dspy_tools[0].args)

if __name__ == "__main__":
    import asyncio

    asyncio.run(run())
```

With the code above, we have successfully collected all available MCP tools and converted
them to DSPy tools.


### Build a DSPy Agent to Handle Customer Requests

Now we will use `dspy.ReAct` to build the agent for handling customer requests. `ReAct` stands
for "reasoning and acting," which asks the LLM to decide whether to call a tool or wrap up the process.
If a tool is required, the LLM takes responsibility for deciding which tool to call and providing
the appropriate arguments.

As usual, we need to create a `dspy.Signature` to define the input and output of our agent:

```python
import dspy

class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent. You are given a list of tools to handle user requests. You should decide the right tool to use in order to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, "
            "e.g., the confirmation_number if it's a flight booking request."
        )
    )
```

And choose an LM for our agent:

```python
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

Then we create the ReAct agent by passing the tools and signature into the `dspy.ReAct` API. We can now
put together the complete code script:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import dspy

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="python",  # Executable
    args=["script_tmp/mcp_server.py"],  # Optional command line arguments
    env=None,  # Optional environment variables
)


class DSPyAirlineCustomerService(dspy.Signature):
    """You are an airline customer service agent. You are given a list of tools to handle user requests.
    You should decide the right tool to use in order to fulfill users' requests."""

    user_request: str = dspy.InputField()
    process_result: str = dspy.OutputField(
        desc=(
            "Message that summarizes the process result, and the information users need, "
            "e.g., the confirmation_number if it's a flight booking request."
        )
    )


dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


async def run(user_request):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            # List available tools
            tools = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            # Create the agent
            react = dspy.ReAct(DSPyAirlineCustomerService, tools=dspy_tools)

            result = await react.acall(user_request=user_request)
            print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run("please help me book a flight from SFO to JFK on 09/01/2025, my name is Adam"))
```

Note that we must call `react.acall` because MCP tools are async by default. Let's execute the script:

```shell
python path_to_your_working_directory/dspy_mcp_agent.py
```

You should see output similar to this:

```
Prediction(
    trajectory={'thought_0': 'I need to fetch flight information for Adam from SFO to JFK on 09/01/2025 to find available flights for booking.', 'tool_name_0': 'fetch_flight_info', 'tool_args_0': {'date': {'year': 2025, 'month': 9, 'day': 1, 'hour': 0}, 'origin': 'SFO', 'destination': 'JFK'}, 'observation_0': ['{"flight_id": "DA123", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 1}, "origin": "SFO", "destination": "JFK", "duration": 3.0, "price": 200.0}', '{"flight_id": "DA125", "date_time": {"year": 2025, "month": 9, "day": 1, "hour": 7}, "origin": "SFO", "destination": "JFK", "duration": 9.0, "price": 500.0}'], ..., 'tool_name_4': 'finish', 'tool_args_4': {}, 'observation_4': 'Completed.'},
    reasoning="I successfully booked a flight for Adam from SFO to JFK on 09/01/2025. I found two available flights, selected the more economical option (flight DA123 at 1 AM for $200), retrieved Adam's user profile, and completed the booking process. The confirmation number for the flight is 8h7clk3q.",
    process_result='Your flight from SFO to JFK on 09/01/2025 has been successfully booked. Your confirmation number is 8h7clk3q.'
)
```

The `trajectory` field contains the entire thinking and acting process. If you're curious about what's happening
under the hood, check out the [Observability Guide](https://dspy.ai/tutorials/observability/) to set up MLflow,
which visualizes every step happening inside `dspy.ReAct`!


## Conclusion

In this guide, we built an airline service agent that utilizes a custom MCP server and the `dspy.ReAct` module. In the context
of MCP support, DSPy provides a simple interface for interacting with MCP tools, giving you the flexibility to implement
any functionality you need.


### tutorials/multihop_search/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/multihop_search/index.ipynb, nbconvert not installed.

### tutorials/observability/index.md

# Tutorial: Debugging and Observability in DSPy

This guide demonstrates how to debug problems and improve observability in DSPy. Modern AI programs often involve multiple components, such as language models, retrievers, and tools. DSPy allows you to build and optimize such complex AI systems in a clean and modular way.

However, as systems grow more sophisticated, the ability to **understand what your system is doing** becomes critical. Without transparency, the prediction process can easily become a black box, making failures or quality issues difficult to diagnose and production maintenance challenging.

By the end of this tutorial, you'll understand how to debug an issue and improve observability using [MLflow Tracing](#tracing). You'll also explore how to build a custom logging solution using callbacks.



## Define a Program

We'll start by creating a simple ReAct agent that uses ColBERTv2's Wikipedia dataset as a retrieval source. You can replace this with a more sophisticated program.

```python
import dspy
from dspy.datasets import HotPotQA

lm = dspy.LM('openai/gpt-4o-mini')
colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(lm=lm, rm=colbert)

agent = dspy.ReAct("question -> answer", tools=[dspy.Retrieve(k=1)])
```

Now, let's ask the agent a simple question:

```python
prediction = agent(question="Which baseball team does Shohei Ohtani play for?")
print(prediction.answer)
```

```
Shohei Ohtani plays for the Los Angeles Angels.
```

Oh, this is incorrect. He no longer plays for the Angels; he moved to the Dodgers and won the World Series in 2024! Let's debug the program and explore potential fixes.

## Using ``inspect_history``

DSPy provides the `inspect_history()` utility, which prints out all LLM invocations made so far:

```python
# Print out 5 LLM calls
dspy.inspect_history(n=5)
```

```
[2024-12-01T10:23:29.144257]

System message:

Your input fields are:
1. `question` (str)

...

Response:

[[ ## Thought_5 ## ]]
The search results continue to be unhelpful and do not provide the current team for Shohei Ohtani in Major League Baseball. I need to conclude that he plays for the Los Angeles Angels based on prior knowledge, as the searches have not yielded updated information.

[[ ## Action_5 ## ]]
Finish[Los Angeles Angels] 

[[ ## completed ## ]]
```
The log reveals that the agent could not retrieve helpful information from the search tool. However, what exactly did the retriever return? While useful, `inspect_history` has some limitations:

* In real-world systems, other components like retrievers, tools, and custom modules play significant roles, but `inspect_history` only logs LLM calls.
* DSPy programs often make multiple LLM calls within a single prediction. Monolith log history makes it hard to organize logs, especially when handling multiple questions.
* Metadata such as parameters, latency, and the relationship between modules are not captured.

**Tracing** addresses these limitations and provides a more comprehensive solution.

## Tracing

[MLflow](https://mlflow.org/docs/latest/llms/tracing/index.html) is an end-to-end machine learning platform that is integrated seamlessly with DSPy to support best practices in LLMOps. Using MLflow's automatic tracing capability with DSPy is straightforward; **No sign up for services or an API key is required**. You just need to install MLflow and call `mlflow.dspy.autolog()` in your notebook or script.

```bash
pip install -U mlflow>=2.18.0
```

```python
import mlflow

mlflow.dspy.autolog()

# This is optional. Create an MLflow Experiment to store and organize your traces.
mlflow.set_experiment("DSPy")
```

Now you're all set! Let's run your agent again:

```python
agent(question="Which baseball team does Shohei Ohtani play for?")
```

MLflow automatically generates a **trace** for the prediction and records it in the experiment. To explore traces visually, launch the MLflow UI by the following command and access it in your browser:

```bash
mlflow ui --port 5000
```

![DSPy MLflow Tracing](./dspy-tracing.gif)

From the retriever step output, you can observe that it returned outdated information; indicating Shohei Ohtani was still playing in the Japanese league and the final answer was based on the LLM's prior knowledge! We should update the dataset or add additional tools to ensure access to the latest information.

!!! info Learn more about MLflow

    MLflow is an end-to-end LLMOps platform that offers extensive features like experiment tracking, evaluation, and deployment. To learn more about DSPy and MLflow integration, visit [this tutorial](../deployment/#deploying-with-mlflow).

For example, we can add a web search capability to the agent, using the [Tavily](https://tavily.com/) web search API.

```python
from dspy.predict.react import Tool
from tavily import TavilyClient

search_client = TavilyClient(api_key="<YOUR_TAVILY_API_KEY>")

def web_search(query: str) -> list[str]:
    """Run a web search and return the content from the top 5 search results"""
    response = search_client.search(query)
    return [r["content"] for r in response["results"]]

agent = dspy.ReAct("question -> answer", tools=[Tool(web_search)])

prediction = agent(question="Which baseball team does Shohei Ohtani play for?")
print(agent.answer)
```

```
Los Angeles Dodgers
```


## Building a Custom Logging Solution

Sometimes, you may want to implement a custom logging solution. For instance, you might need to log specific events triggered by a particular module. DSPy's **callback** mechanism supports such use cases. The ``BaseCallback`` class provides several handlers for customizing logging behavior:

|Handlers|Description|
|:--|:--|
|`on_module_start` / `on_module_end` | Triggered when a `dspy.Module` subclass is invoked. |
|`on_lm_start` / `on_lm_end` | Triggered when a `dspy.LM` subclass is invoked. |
|`on_adapter_format_start` / `on_adapter_format_end`| Triggered when a `dspy.Adapter` subclass formats the input prompt. |
|`on_adapter_parse_start` / `on_adapter_parse_end`| Triggered when a `dspy.Adapter` subclass postprocess the output text from an LM. |

Here’s an example of custom callback that logs the intermediate steps of a ReAct agent:

```python
import dspy
from dspy.utils.callback import BaseCallback

# 1. Define a custom callback class that extends BaseCallback class
class AgentLoggingCallback(BaseCallback):

    # 2. Implement on_module_end handler to run a custom logging code.
    def on_module_end(self, call_id, outputs, exception):
        step = "Reasoning" if self._is_reasoning_output(outputs) else "Acting"
        print(f"== {step} Step ===")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\n")

    def _is_reasoning_output(self, outputs):
        return any(k.startswith("Thought") for k in outputs.keys())

# 3. Set the callback to DSPy setting so it will be applied to program execution
dspy.configure(callbacks=[AgentLoggingCallback()])
```


```
== Reasoning Step ===
  Thought_1: I need to find the current team that Shohei Ohtani plays for in Major League Baseball.
  Action_1: Search[Shohei Ohtani current team 2023]

== Acting Step ===
  passages: ["Shohei Ohtani ..."]

...
```

!!! info Handling Inputs and Outputs in Callbacks

    Be cautious when working with input or output data in callbacks. Mutating them in-place can modify the original data passed to the program, potentially leading to unexpected behavior. To avoid this, it’s strongly recommended to create a copy of the data before performing any operations that may alter it.


### tutorials/optimize_ai_program/index.md

- [Math Reasoning](/tutorials/math/)
- [Classification Finetuning](/tutorials/classification_finetuning/)
- [Advanced Tool Use](/tutorials/tool_use/)
- [Finetuning Agents](/tutorials/games/)

### tutorials/optimizer_tracking/index.md

# Tracking DSPy Optimizers with MLflow

This tutorial demonstrates how to use MLflow to track and analyze your DSPy optimization process. MLflow's built-in integration for DSPy provides traceability and debuggability for your DSPy optimization experience. It allows you to understand the intermediate trials during the optimization, store the optimized program and its results, and provides observability into your program execution.

Through the autologging capability, MLflow tracks the following information:

* **Optimizer Parameters**
    * Number of few-shot examples
    * Number of candidates
    * Other configuration settings

* **Program States**
    * Initial instuctions and few-shot examples
    * Optimized instuctions and few-shot examples
    * Intermediate instuctions and few-shot examples during optimization

* **Datasets**
    * Training data used
    * Evaluation data used

* **Performance Progression**
    * Overall metric progression
    * Performance at each evaluation step

* **Traces**
    * Program execution traces
    * Model responses
    * Intermediate prompts

## Getting Started

### 1. Install MLflow
First, install MLflow (version 2.21.1 or later):

```bash
pip install mlflow>=2.21.1
```

### 2. Start MLflow Tracking Server

Let's spin up the MLflow tracking server with the following command. This will start a local server at `http://127.0.0.1:5000/`:

```bash
# It is highly recommended to use SQL store when using MLflow tracing
mlflow server --backend-store-uri sqlite:///mydb.sqlite
```

### 3. Enable Autologging

Configure MLflow to track your DSPy optimization:

```python
import mlflow
import dspy

# Enable autologging with all features
mlflow.dspy.autolog(
    log_compiles=True,    # Track optimization process
    log_evals=True,       # Track evaluation results
    log_traces_from_compile=True  # Track program traces during optimization
)

# Configure MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")  # Use local MLflow server
mlflow.set_experiment("DSPy-Optimization")
```

### 4. Optimizing Your Program

Here's a complete example showing how to track the optimization of a math problem solver:

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

# Configure your language model
lm = dspy.LM(model="openai/gpt-4o")
dspy.configure(lm=lm)

# Load dataset
gsm8k = GSM8K()
trainset, devset = gsm8k.train, gsm8k.dev

# Define your program
program = dspy.ChainOfThought("question -> answer")

# Create and run optimizer with tracking
teleprompter = dspy.teleprompt.MIPROv2(
    metric=gsm8k_metric,
    auto="light",
)

# The optimization process will be automatically tracked
optimized_program = teleprompter.compile(
    program,
    trainset=trainset,
)
```

### 5. Viewing Results

Once your optimization is complete, you can analyze the results through MLflow's UI. Let's walk through how to explore your optimization runs.

#### Step 1: Access the MLflow UI
Navigate to `http://localhost:5000` in your web browser to access the MLflow tracking server UI.

#### Step 2: Understanding the Experiment Structure
When you open the experiment page, you'll see a hierarchical view of your optimization process. The parent run represents your overall optimization process, while the child runs show each intermediate version of your program that was created during optimization.

![Experiments](./experiment.png)

#### Step 3: Analyzing the Parent Run
Clicking on the parent run reveals the big picture of your optimization process. You'll find detailed information about your optimizer's configuration parameters and how your evaluation metrics progressed over time. The parent run also stores your final optimized program, including the instructions, signature definitions, and few-shot examples that were used. Additionally, you can review the training data that was used during the optimization process.

![Parent Run](./parent_run.png)

#### Step 4: Examining Child Runs
Each child run provides a detailed snapshot of a specific optimization attempt. When you select a child run from the experiment page, you can explore several aspects of that particular intermediate program.
On the run parameter tab or artifact tab, you can review the instructions and few-shot examples used for the intermediate program.
One of the most powerful features is the Traces tab, which provides a step-by-step view of your program's execution. Here you can understand exactly how your DSPy program processes inputs and generates outputs.

![Child Run](./child_run.png)

### 6. Loading Models for Inference
You can load the optimized program directly from the MLflow tracking server for inference:

```python
model_path = mlflow.artifacts.download_artifacts("mlflow-artifacts:/path/to/best_model.json")
program.load(model_path)
```

## Troubleshooting

- If traces aren't appearing, ensure `log_traces_from_compile=True`
- For large datasets, consider setting `log_traces_from_compile=False` to avoid memory issues
- Use `mlflow.get_run(run_id)` to programmatically access MLflow run data

For more features, explore the [MLflow Documentation](https://mlflow.org/docs/latest/llms/dspy).


### tutorials/output_refinement/best-of-n-and-refine.md

# Output Refinement: BestOfN and Refine

Both `BestOfN` and `Refine` are DSPy modules designed to improve the reliability and quality of predictions by making multiple `LM` calls with different parameter settings. Both modules stop when they have reached `N` attempts or when the `reward_fn` returns an award above the `threshold`.

## BestOfN

`BestOfN` is a module that runs the provided module multiple times (up to `N`) with different temperature settings. It returns either the first prediction that passes a specified threshold or the one with the highest reward if none meets the threshold.

### Basic Usage

Lets say we wanted to have the best chance of getting a one word answer from the model. We could use `BestOfN` to try multiple temperature settings and return the best result.

```python
import dspy

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

best_of_3 = dspy.BestOfN(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = best_of_3(question="What is the capital of Belgium?")
print(result.answer)  # Brussels
```

### Error Handling

By default, if the module encounters an error during an attempt, it will continue trying until it reaches `N` attempts. You can adjust this behavior with the `fail_count` parameter:

```python
best_of_3 = dspy.BestOfN(
    module=qa, 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0,
    fail_count=1
)

best_of_3(question="What is the capital of Belgium?")
# raises an error after the first failure
```

## Refine

`Refine` extends the functionality of `BestOfN` by adding an automatic feedback loop. After each unsuccessful attempt (except the final one), it automatically generates detailed feedback about the module's performance and uses this feedback as hints for subsequent runs.

### Basic Usage

```python
import dspy

def one_word_answer(args, pred: dspy.Prediction) -> float:
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

refine = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"), 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0
)

result = refine(question="What is the capital of Belgium?")
print(result.answer)  # Brussels
```

### Error Handling

Like `BestOfN`, `Refine` will try up to `N` times by default, even if errors occur. You can control this with the `fail_count` parameter:

```python
# Stop after the first error
refine = dspy.Refine(
    module=qa, 
    N=3, 
    reward_fn=one_word_answer, 
    threshold=1.0,
    fail_count=1
)
```

## Comparison: BestOfN vs. Refine

Both modules serve similar purposes but differ in their approach:

- `BestOfN` simply tries different temperature settings and selects the best resulting prediction as defined by the `reward_fn`.
- `Refine` adds an feedback loop, using the lm to generate a detailed feedback about the module's own performance using the previous prediction and the code in the `reward_fn`. This feedback is then used as hints for subsequent runs.

## Practical Examples

### Ensuring Factual Correctness

```python
import dspy

class FactualityJudge(dspy.Signature):
    """Determine if a statement is factually accurate."""
    statement: str = dspy.InputField()
    is_factual: bool = dspy.OutputField()

factuality_judge = dspy.ChainOfThought(FactualityJudge)

def factuality_reward(args, pred: dspy.Prediction) -> float:
    statement = pred.answer    
    result = factuality_judge(statement)    
    return 1.0 if result.is_factual else 0.0

refined_qa = dspy.Refine(
    module=dspy.ChainOfThought("question -> answer"),
    N=3,
    reward_fn=factuality_reward,
    threshold=1.0
)

result = refined_qa(question="Tell me about Belgium's capital city.")
print(result.answer)
```

### Summarization - Controlling Response Length

```python
import dspy

def ideal_length_reward(args, pred: dspy.Prediction) -> float:
    """
    Reward the summary for being close to 75 words with a tapering off for longer summaries.
    """
    word_count = len(pred.summary.split())
    distance = abs(word_count - 75)
    return max(0.0, 1.0 - (distance / 125))

optimized_summarizer = dspy.BestOfN(
    module=dspy.ChainOfThought("text -> summary"),
    N=50,
    reward_fn=ideal_length_reward,
    threshold=0.9
)

result = optimized_summarizer(
    text="[Long text to summarize...]"
)
print(result.summary)
```

## Migration from `dspy.Suggest` and `dspy.Assert`

`BestOfN` and `Refine` are the replacements for `dspy.Suggest` and `dspy.Assert` as of DSPy 2.6.


### tutorials/papillon/index.md

Please refer to [this tutorial from the PAPILLON authors](https://colab.research.google.com/github/Columbia-NLP-Lab/PAPILLON/blob/main/papillon_tutorial.ipynb) using DSPy.

This tutorial demonstrates a few aspects of using DSPy in a more advanced context:

1. It builds a multi-stage `dspy.Module` that involves a small local LM using an external tool.
2. It builds a multi-stage _judge_ in DSPy, and uses it as a metric for evaluation.
3. It uses this judge for optimizing the `dspy.Module`, using a large model as a teacher for a small local LM.


### tutorials/program_of_thought/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/program_of_thought/index.ipynb, nbconvert not installed.

### tutorials/rag/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/rag/index.ipynb, nbconvert not installed.

### tutorials/rl_ai_program/index.md

See the links on the side bar.

### tutorials/rl_multihop/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/rl_multihop/index.ipynb, nbconvert not installed.

### tutorials/rl_papillon/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/rl_papillon/index.ipynb, nbconvert not installed.

### tutorials/saving/index.md

# Tutorial: Saving and Loading your DSPy program

This guide demonstrates how to save and load your DSPy program. At a high level, there are two ways to save your DSPy program:

1. Save the state of the program only, similar to weights-only saving in PyTorch.
2. Save the whole program, including both the architecture and the state, which is supported by `dspy>=2.6.0`.

## State-only Saving

State represents the DSPy program's internal state, including the signature, demos (few-shot examples), and other information like
the `lm` to use for each `dspy.Predict` in the program. It also includes configurable attributes of other DSPy modules like
`k` for `dspy.retrievers.Retriever`. To save the state of a program, use the `save` method and set `save_program=False`. You can
choose to save the state to a JSON file or a pickle file. We recommend saving the state to a JSON file because it is safer and readable.
But sometimes your program contains non-serializable objects like `dspy.Image` or `datetime.datetime`, in which case you should save
the state to a pickle file.

Let's say we have compiled a program with some data, and we want to save the program for future usage:

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))

gsm8k = GSM8K()
gsm8k_trainset = gsm8k.train[:10]
dspy_program = dspy.ChainOfThought("question -> answer")

optimizer = dspy.BootstrapFewShot(metric=gsm8k_metric, max_bootstrapped_demos=4, max_labeled_demos=4, max_rounds=5)
compiled_dspy_program = optimizer.compile(dspy_program, trainset=gsm8k_trainset)
```

To save the state of your program to json file:

```python
compiled_dspy_program.save("./dspy_program/program.json", save_program=False)
```

To save the state of your program to a pickle file:

```python
compiled_dspy_program.save("./dspy_program/program.pkl", save_program=False)
```

To load your saved state, you need to **recreate the same program**, then load the state using the `load` method.

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
loaded_dspy_program.load("./dspy_program/program.json")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

Or load the state from a pickle file:

```python
loaded_dspy_program = dspy.ChainOfThought("question -> answer") # Recreate the same program.
loaded_dspy_program.load("./dspy_program/program.pkl")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

## Whole Program Saving

Starting from `dspy>=2.6.0`, DSPy supports saving the whole program, including the architecture and the state. This feature
is powered by `cloudpickle`, which is a library for serializing and deserializing Python objects.

To save the whole program, use the `save` method and set `save_program=True`, and specify a directory path to save the program
instead of a file name. We require a directory path because we also save some metadata, e.g., the dependency versions along
with the program itself.

```python
compiled_dspy_program.save("./dspy_program/", save_program=True)
```

To load the saved program, directly use `dspy.load` method:

```python
loaded_dspy_program = dspy.load("./dspy_program/")

assert len(compiled_dspy_program.demos) == len(loaded_dspy_program.demos)
for original_demo, loaded_demo in zip(compiled_dspy_program.demos, loaded_dspy_program.demos):
    # Loaded demo is a dict, while the original demo is a dspy.Example.
    assert original_demo.toDict() == loaded_demo
assert str(compiled_dspy_program.signature) == str(loaded_dspy_program.signature)
```

With whole program saving, you don't need to recreate the program, but can directly load the architecture along with the state.
You can pick the suitable saving approach based on your needs.

### Serializing Imported Modules

When saving a program with `save_program=True`, you might need to include custom modules that your program depends on. This is
necessary if your program depends on these modules, but at loading time these modules are not imported before calling `dspy.load`.

You can specify which custom modules should be serialized with your program by passing them to the `modules_to_serialize`
parameter when calling `save`. This ensures that any dependencies your program relies on are included during serialization and
available when loading the program later.

Under the hood this uses cloudpickle's `cloudpickle.register_pickle_by_value` function to register a module as picklable by value.
When a module is registered this way, cloudpickle will serialize the module by value rather than by reference, ensuring that the
module contents are preserved with the saved program.

For example, if your program uses custom modules:

```python
import dspy
import my_custom_module

compiled_dspy_program = dspy.ChainOfThought(my_custom_module.custom_signature)

# Save the program with the custom module
compiled_dspy_program.save(
    "./dspy_program/",
    save_program=True,
    modules_to_serialize=[my_custom_module]
)
```

This ensures that the required modules are properly serialized and available when loading the program later. Any number of
modules can be passed to `modules_to_serialize`. If you don't specify `modules_to_serialize`, no additional modules will be
registered for serialization.

## Backward Compatibility

As of `dspy<2.7`, we don't guarantee the backward compatibility of the saved program. For example, if you save the program with `dspy==2.5.35`,
at loading time please make sure to use the same version of DSPy to load the program, otherwise the program may not work as expected. Chances
are that loading a saved file in a different version of DSPy will not raise an error, but the performance could be different from when
the program was saved.

Starting from `dspy>=2.7`, we will guarantee the backward compatibility of the saved program in major releases, i.e., programs saved in `dspy==2.7.0`
should be loadable in `dspy==2.7.10`.


### tutorials/streaming/index.md

# Streaming

In this guide, we will walk you through how to enable streaming in your DSPy program. DSPy Streaming
consists of two parts:

- **Output Token Streaming**: Stream individual tokens as they're generated, rather than waiting for the complete response.
- **Intermediate Status Streaming**: Provide real-time updates about the program's execution state (e.g., "Calling web search...", "Processing results...").

## Output Token Streaming

DSPy's token streaming feature works with any module in your pipeline, not just the final output. The only requirement is that the streamed field must be of type `str`. To enable token streaming:

1. Wrap your program with `dspy.streamify`
2. Create one or more `dspy.streaming.StreamListener` objects to specify which fields to stream

Here's a basic example:

```python
import os

import dspy

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

# Enable streaming for the 'answer' field
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)
```

To consume the streamed output:

```python
import asyncio

async def read_output_stream():
    output_stream = stream_predict(question="Why did a chicken cross the kitchen?")

    async for chunk in output_stream:
        print(chunk)

asyncio.run(read_output_stream())
```

This will produce output like:

```
StreamResponse(predict_name='self', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' other')
StreamResponse(predict_name='self', signature_field_name='answer', chunk=' side of the frying pan!')
Prediction(
    answer='To get to the other side of the frying pan!'
)
```

Note: Since `dspy.streamify` returns an async generator, you must use it within an async context. If you're using an environment like Jupyter or Google Colab that already has an event loop (async context), you can use the generator directly.

You may have noticed that the above streaming contains two different entities: `StreamResponse`
and `Prediction.` `StreamResponse` is the wrapper over streaming tokens on the field being listened to, and in
this example it is the `answer` field. `Prediction` is the program's final output. In DSPy, streaming is
implemented in a sidecar fashion: we enable streaming on the LM so that LM outputs a stream of tokens. We send these
tokens to a side channel, which is being continuously read by the user-defined listeners. Listeners keep interpreting
the stream, and decides if the `signature_field_name` it is listening to has started to appear and has finalized.
Once it decides that the field appears, the listener begins outputing tokens to the async generator users can
read. Listeners' internal mechanism changes according to the adapter behind the scene, and because usually
we cannot decide if a field has finalized until seeing the next field, the listener buffers the output tokens
before sending to the final generator, which is why you will usually see the last chunk of type `StreamResponse`
has more than one token. The program's output is also written to the stream, which is the chunk of `Prediction`
as in the sample output above.

To handle these different types and implement custom logic:

```python
import asyncio

async def read_output_stream():
  output_stream = stream_predict(question="Why did a chicken cross the kitchen?")

  async for chunk in output_stream:
    return_value = None
    if isinstance(chunk, dspy.streaming.StreamResponse):
      print(f"Output token of field {chunk.signature_field_name}: {chunk.chunk}")
    elif isinstance(chunk, dspy.Prediction):
      return_value = chunk


program_output = asyncio.run(read_output_stream())
print("Final output: ", program_output)
```

### Understand `StreamResponse`

`StreamResponse` (`dspy.streaming.StreamResponse`) is the wrapper class of streaming tokens. It comes with 3
fields:

- `predict_name`: the name of the predict that holds the `signature_field_name`. The name is the
  same name of keys as you run `your_program.named_predictors()`. In the code above because `answer` is from
  the `predict` itself, so the `predict_name` shows up as `self`, which is the only key as your run
  `predict.named_predictors()`.
- `signature_field_name`: the output field that these tokens map to. `predict_name` and `signature_field_name`
  together form the unique identifier of the field. We will demonstrate how to handle multiple fields streaming
  and duplicated field name later in this guide.
- `chunk`: the value of the stream chunk.

### Streaming with Cache

When a cached result is found, the stream will skip individual tokens and only yield the final `Prediction`. For example:

```
Prediction(
    answer='To get to the other side of the dinner plate!'
)
```

### Streaming Multiple Fields

You can monitor multiple fields by creating a `StreamListener` for each one. Here's an example with a multi-module program:

```python
import asyncio

import dspy

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.settings.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.predict1 = dspy.Predict("question->answer")
        self.predict2 = dspy.Predict("answer->simplified_answer")

    def forward(self, question: str, **kwargs):
        answer = self.predict1(question=question)
        simplified_answer = self.predict2(answer=answer)
        return simplified_answer


predict = MyModule()
stream_listeners = [
    dspy.streaming.StreamListener(signature_field_name="answer"),
    dspy.streaming.StreamListener(signature_field_name="simplified_answer"),
]
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
)

async def read_output_stream():
    output = stream_predict(question="why did a chicken cross the kitchen?")

    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value

program_output = asyncio.run(read_output_stream())
print("Final output: ", program_output)
```

The output will look like:

```
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' other side of the recipe!')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk='To')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' reach')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' the')
StreamResponse(predict_name='predict2', signature_field_name='simplified_answer', chunk=' other side of the recipe!')
Final output:  Prediction(
    simplified_answer='To reach the other side of the recipe!'
)
```

#### Handling Duplicate Field Names

When streaming fields with the same name from different modules, specify both the `predict` and `predict_name` in the `StreamListener`:

```python
import asyncio

import dspy

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.settings.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.predict1 = dspy.Predict("question->answer")
        self.predict2 = dspy.Predict("question, answer->answer, score")

    def forward(self, question: str, **kwargs):
        answer = self.predict1(question=question)
        simplified_answer = self.predict2(answer=answer)
        return simplified_answer


predict = MyModule()
stream_listeners = [
    dspy.streaming.StreamListener(
        signature_field_name="answer",
        predict=predict.predict1,
        predict_name="predict1"
    ),
    dspy.streaming.StreamListener(
        signature_field_name="answer",
        predict=predict.predict2,
        predict_name="predict2"
    ),
]
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
)


async def read_output_stream():
    output = stream_predict(question="why did a chicken cross the kitchen?")

    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
    return return_value


program_output = asyncio.run(read_output_stream())
print("Final output: ", program_output)
```

The output will be like:

```
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk='To')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' get')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' the')
StreamResponse(predict_name='predict1', signature_field_name='answer', chunk=' other side of the recipe!')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk="I'm")
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' ready')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' to')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' assist')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk=' you')
StreamResponse(predict_name='predict2', signature_field_name='answer', chunk='! Please provide a question.')
Final output:  Prediction(
    answer="I'm ready to assist you! Please provide a question.",
    score='N/A'
)
```

## Intermediate Status Streaming

Status streaming keeps users informed about the program's progress, especially useful for long-running operations like tool calls or complex AI pipelines. To implement status streaming:

1. Create a custom status message provider by subclassing `dspy.streaming.StatusMessageProvider`
2. Override the desired hook methods to provide custom status messages
3. Pass your provider to `dspy.streamify`

Example:

```python
class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return f"Calling LM with inputs {inputs}..."

    def lm_end_status_message(self, outputs):
        return f"Tool finished with output: {outputs}!"
```

Available hooks:

- lm_start_status_message: status message at the start of calling dspy.LM.
- lm_end_status_message: status message at the end of calling dspy.LM.
- module_start_status_message: status message at the start of calling a dspy.Module.
- module_end_status_message: status message at the start of calling a dspy.Module.
- tool_start_status_message: status message at the start of calling dspy.Tool.
- tool_end_status_message: status message at the end of calling dspy.Tool.

Each hook should return a string containing the status message.

After creating the message provider, just pass it to `dspy.streamify`, and you can enable both
status message streaming and output token streaming. Please see the example below. The intermediate
status message is represented in the class `dspy.streaming.StatusMessage`, so we need to have
another condition check to capture it.

```python
import asyncio

import dspy

lm = dspy.LM("openai/gpt-4o-mini", cache=False)
dspy.settings.configure(lm=lm)


class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()

        self.tool = dspy.Tool(lambda x: 2 * x, name="double_the_number")
        self.predict = dspy.ChainOfThought("num1, num2->sum")

    def forward(self, num, **kwargs):
        num2 = self.tool(x=num)
        return self.predict(num1=num, num2=num2)


class MyStatusMessageProvider(dspy.streaming.StatusMessageProvider):
    def tool_start_status_message(self, instance, inputs):
        return f"Calling Tool {instance.name} with inputs {inputs}..."

    def tool_end_status_message(self, instance, outputs):
        return f"Tool finished with output: {outputs}!"


predict = MyModule()
stream_listeners = [
    # dspy.ChainOfThought has a built-in output field called "reasoning".
    dspy.streaming.StreamListener(signature_field_name="reasoning"),
]
stream_predict = dspy.streamify(
    predict,
    stream_listeners=stream_listeners,
)


async def read_output_stream():
    output = stream_predict(num=3)

    return_value = None
    async for chunk in output:
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk)
        elif isinstance(chunk, dspy.Prediction):
            return_value = chunk
        elif isinstance(chunk, dspy.streaming.StatusMessage):
            print(chunk)
    return return_value


program_output = asyncio.run(read_output_stream())
print("Final output: ", program_output)
```

Sample output:

```
StatusMessage(message='Calling tool double_the_number...')
StatusMessage(message='Tool calling finished! Querying the LLM with tool calling results...')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='To')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' find')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' the')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' sum')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' of')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' the')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' two')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' numbers')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=',')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' we')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' simply')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' add')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' them')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' together')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='.')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' Here')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=',')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' ')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk='3')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' plus')
StreamResponse(predict_name='predict.predict', signature_field_name='reasoning', chunk=' 6 equals 9.')
Final output:  Prediction(
    reasoning='To find the sum of the two numbers, we simply add them together. Here, 3 plus 6 equals 9.',
    sum='9'
)
```

## Synchronous Streaming

By default calling a streamified DSPy program produces an async generator. In order to get back
a sync generator, you can set the flag `async_streaming=False`:


```python
import os

import dspy

os.environ["OPENAI_API_KEY"] = "your_api_key"

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

predict = dspy.Predict("question->answer")

# Enable streaming for the 'answer' field
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    async_streaming=False,
)

output = stream_predict(question="why did a chicken cross the kitchen?")

program_output = None
for chunk in output:
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk)
    elif isinstance(chunk, dspy.Prediction):
        program_output = chunk
print(f"Program output: {program_output}")
```

### tutorials/tool_use/index.ipynb

Cannot convert /workspace/dspy/docs/docs/tutorials/tool_use/index.ipynb, nbconvert not installed.
