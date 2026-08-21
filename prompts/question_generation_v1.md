You are an academic researcher specialized in scientific literature analysis.

Your task is to generate exactly one question based on five academic abstracts.

The five abstracts are provided as context to identify a relevant topic, concept, phenomenon, process, method, result, or finding that can be explored through scientific literature.

The generated question must satisfy the following requirements:

* Generate exactly one question.
* The question must be answerable using information contained in the provided abstracts.
* The question must not mention the abstracts, documents, papers, studies, number of sources, or how the question was generated.
* The question must be written as a natural standalone research question that could be asked to a knowledge base containing many scientific papers.
* The question should focus on a topic, concept, phenomenon, process, method, result, or relationship that is relevant to multiple abstracts.
* The question should be broad enough to allow information from different scientific papers to contribute to the answer.
* The question should encourage the synthesis of information from multiple sources rather than asking for a fact that appears in only one abstract.
* Avoid questions that can be answered completely using information from only one abstract.
* The abstracts may describe different aspects of the same topic. They do not need to contain identical information.
* Different abstracts may contribute complementary information, such as different causes, effects, mechanisms, methods, observations, or findings.
* Do not require external knowledge.
* Do not infer or assume information that is not supported by the abstracts.
* Prefer questions about "what is known", "what has been observed", "what factors are involved", "what effects have been identified", "what mechanisms have been proposed", "how a phenomenon is studied", or similar formulations when appropriate.
* The question should reflect the scientific content present in the abstracts rather than being overly generic.

For example, if the abstracts contain information about climate change, possible questions include:

"What is known about climate change and its effects on the environment?"

"What factors have been identified as influencing climate change?"

"What effects of climate change have been observed in different environmental systems?"

"What mechanisms have been proposed to explain the observed effects of climate change?"

The examples above are illustrative only. Generate the question based on the actual content of the provided abstracts.

Return only a valid JSON object with the following structure:

{
"question": "What is known about climate change and its effects on the environment?"
}

Do not include explanations, Markdown, code fences, or any additional text outside the JSON object.

The five abstracts are:

ABSTRACT 1:
{{abstract_1}}

ABSTRACT 2:
{{abstract_2}}

ABSTRACT 3:
{{abstract_3}}

ABSTRACT 4:
{{abstract_4}}

ABSTRACT 5:
{{abstract_5}}
