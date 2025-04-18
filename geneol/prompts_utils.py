from collections import defaultdict
import copy
import numpy as np




def get_sum_prompt_fs(input_text):
    text = f"Summarize the input sentence while preserving the exact meaning of the sentence. Do not output any additional explanation. Only output the summary.\nSentence: {input_text}"
    messages = [
    {"role": "user", "content": text},
    ]

    return messages






def get_pos_prompt(version, input_text):
    if(version==0):
        pos_template = pos0_template
        pos_template_examples = pos0_template_examples
    elif(version==1):
        pos_template = pos1_template
        pos_template_examples = pos1_template_examples
    elif(version==2):
        pos_template = pos2_template
        pos_template_examples = pos2_template_examples
    elif(version==3):
        pos_template = pos3_template
        pos_template_examples = pos3_template_examples
    elif(version==4):
        pos_template = pos4_template
        pos_template_examples = pos4_template_examples
    else:
        assert False, ("version is", version)

    pos_template = copy.deepcopy(pos_template)
    assert len(pos_template_examples)==10
    demos_ids = np.random.choice(10, 5, replace=False)
    text_list = [each for demo_id in demos_ids for each in pos_template_examples[demo_id]]+[input_text]

    
    for u in range(0, len(text_list), 2):
        pos_template[2+u]["content"] = text_list[u]
        assert pos_template[2+u]["role"] == "user"

    for a in range(1, len(text_list), 2):
        pos_template[2+a]["content"] = text_list[a]
        assert pos_template[2+a]["role"] == "assistant"
    
    return pos_template

def get_neg_prompt(version, input_text):
    if(version==1):
        neg_template = neg1_template
        neg_template_examples = neg1_template_examples
    elif(version==2):
        neg_template = neg2_template
        neg_template_examples = neg2_template_examples
    elif(version==3):
        neg_template = neg3_template
        neg_template_examples = neg3_template_examples
    elif(version==4):
        neg_template = neg4_template
        neg_template_examples = neg4_template_examples
    else:
        assert False, ("version is", version)

    neg_template = copy.deepcopy(neg_template)
    demos_ids = np.random.choice(10, 5, replace=False)
    text_list = [each for demo_id in demos_ids for each in neg_template_examples[demo_id]]+[input_text]
    for u in range(0, len(text_list), 2):
        neg_template[2+u]["content"] = text_list[u]
        assert neg_template[2+u]["role"] == "user"

    for a in range(1, len(text_list), 2):
        neg_template[2+a]["content"] = text_list[a]
        assert neg_template[2+a]["role"] == "assistant"
    
    return neg_template



def get_diverse_prompt_fs(input_text):
    """
    FS stands for 'Few-Shot'. This function generates a prompt for producing diverse outputs 
    based on a given input text using few-shot learning techniques.
    """
    text = f"""Make changes to the following sentence to output 10 diverse sentences. Following are some directions for the changes.
1. Paraphrase the input sentence or phrase, providing an alternative expression with the same meaning.
2. Rewrite the input sentence or phrase using different sentence structure and different words while preserving its original meaning.
3. Create a sentence or phrase that is also true, assuming the provided input sentence or phrase is true.
4. Provide a concise paraphrase of the input sentence or phrase, maintaining the core meaning while altering the words and sentence structure. Feel free to omit some of the non-essential details like adjectives or adverbs.

With these possible transformations in mind, transform the following sentence. Output the 10 transformed sentences in 10 new lines. Do not output any explanation. Do not output anything else.
Sentence: {input_text}"""

    messages = [
    {"role": "user", "content": text},
    ]

    return messages



''' 
EOL prompt:
{}
that was not good, give me a better one {} 
{} 



User: The sentence "[Text]" in one word means:"
Assistant: [Mask]
User: This doesn't entirely represent the sentence. Try again. The sentence "[Text]" in one word means:
Assistant: [Mask]
User: This doesn't entirely represent the sentence. Try again. The sentence "[Text]" in one word means:
Assistant: [Mask]

'''


def get_diverse_prompt2(input_text, task):
    text = f"Make changes to the following sentence to output 10 diverse sentences. Following are some directions for the changes.\n1. Paraphrase the input sentence or phrase, providing an alternative expression with the same meaning.\n2. Rewrite the input sentence or phrase using different sentence structure and different words while preserving its original meaning.\n3. Create a sentence or phrase that is also true, assuming the provided input sentence or phrase is true.\n4. Provide a concise paraphrase of the input sentence or phrase, maintaining the core meaning while altering the words and sentence structure. Feel free to omit some of the non-essential details like adjectives or adverbs.\n\nWith these possible transformations in mind, transform the following sentence. Do not output line number. Do not output any explanation. Output should be formatted as a json string. The sentences should be in a list with key 'generations'. You need to return 10 sentences in the list.\nSentence: {input_text}"
    messages = [
    {"role": "user", "content": text},
    ]

    return messages


def get_task_specific_emb_prompt(input_text, task):
    # Define a dictionary mapping tasks to their respective prompt templates
    task_prompts = defaultdict(lambda: f"<s>[INST] What single word best captures the core meaning of this sentence: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]")
    # Add task-specific prompts
    task_prompts.update({
        "MR": f"<s>[INST] What single word best captures the SENTIMENT (positive/negative) of this movie review: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "CR": f"<s>[INST] What single word best captures the SENTIMENT (positive/negative) of this product review: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "SUBJ": f"<s>[INST] Is this movie review SUBJECTIVE (opinion-based) or OBJECTIVE (fact-based)? \"{input_text}\". Reply with ONLY ONE WORD. [/INST]",
        "MPQA": f"<s>[INST] What single word best captures the SENTIMENT (positive/negative) of this text: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "SST": f"<s>[INST] What single word best captures the SENTIMENT (positive/negative) of this review: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "TREC": f"<s>[INST] Categorize this question: \"{input_text}\". Choose ONE category: Description, Entity, Abbreviation, Person, Quantity, or Location. Reply with ONLY ONE WORD. [/INST]",
        "MRPC": f"<s>[INST] Are these sentences paraphrases of each other? \"{input_text}\". Reply with ONLY 'Yes' or 'No'. [/INST]",
        "MedrxivClusteringS2S": f"<s>[INST] What single CATEGORY best describes this medical paper title: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "TwentyNewsgroupsClustering": f"<s>[INST] What single TOPIC best describes this news article: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "AmazonCounterfactualClassification": f"<s>[INST] Is this Amazon review COUNTERFACTUAL or NOT? \"{input_text}\". Reply with ONLY ONE WORD. [/INST]",
        "Banking77Classification": f"<s>[INST] What single word best describes the INTENT of this banking query: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "EmotionClassification": f"<s>[INST] Which EMOTION is expressed in this tweet: \"{input_text}\"? Choose ONE: anger, fear, joy, love, sadness, surprise. Reply with ONLY ONE WORD. [/INST]",
        "AskUbuntuDupQuestions": f"<s>[INST] What single word best describes the TOPIC of this Ubuntu question: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "SciDocsRR": f"<s>[INST] What single word best captures the core TOPIC of this scientific paper title: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "StackOverflowDupQuestions": f"<s>[INST] What single word best describes the PROGRAMMING TOPIC of this Stack Overflow question: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "SprintDuplicateQuestions": f"<s>[INST] What single word best describes the TOPIC of this Sprint forum question: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "TwitterSemEval2015": f"<s>[INST] What single word best captures the core TOPIC of this tweet: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
        "TwitterURLCorpus": f"<s>[INST] What single word best summarizes what this tweet is about: \"{input_text}\"? Reply with ONLY ONE WORD. [/INST]",
    })

    # Return the appropriate task-specific prompt
    return task_prompts[task]

def get_task_specific_gen_prompt(input_text, task):
    """
    Placeholder function to fix the error in the code.
    This function is called in the r5 method but was missing from the codebase.
    
    Args:
        input_text (str): The input sentence
        task (str): The task name
        
    Returns:
        dict: A message dictionary for prompt engineering
    """
    # For RegenerateEOL, we don't actually need to generate sentence transformations
    # We'll just return a placeholder prompt to avoid errors
    return {"role": "user", "content": f"Considering the sentence: \"{input_text}\", provide a one-word summary."}

def get_regenerate_emb_prompts(input_text, num_diverse, task=None, previous_embeddings=None):
    """
    Generate a sequence of prompts for RegenerateEOL using the EOL approach.
    
    Args:
        input_text (str): The input sentence to embed
        num_diverse (int): Number of diverse embeddings to generate (including original)
        task (str, optional): The task context
        previous_embeddings (list, optional): List of previous embedding words to guide diversity
        
    Returns:
        list: A list of prompt strings for generating diverse embeddings
    """
    # Initialize prompts list
    prompts = []
    
    # Standard EOL prompt format - this is the core prompt template for all embeddings
    if previous_embeddings is None or len(previous_embeddings) == 0:
        # First prompt with no previous context
        base_prompt = f"<s>The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind, this sentence: \"{input_text}\" means in one word:"
    else:
        # Include previous embeddings as context for diversity
        previous_words = ", ".join([f'"{word}"' for word in previous_embeddings])
        base_prompt = f"<s>The essence of a sentence is often captured by its main subjects and actions, while descriptive terms such as {previous_words} provide additional but less central details. With this in mind, this sentence: \"{input_text}\" means in one word:"
    
    # Add the prompt to the list
    prompts.append(base_prompt)
    
    # Return the prompt
    return prompts

def get_regenerate_conversation_prompt(input_text, num_diverse, task=None):
    """
    Generate a conversation-style prompt for RegenerateEOL that simulates asking for diverse embeddings.
    
    Args:
        input_text (str): The input sentence to embed
        num_diverse (int): Number of diverse embeddings to generate (including original)
        task (str, optional): The task context
        
    Returns:
        list: A list of message dictionaries for the chat template
    """
    # Base task prompt
    task_prefix = ""
    if task is not None:
        task_mapping = {
            "MR": "In this task, you're given a movie review, and you need to classify its sentiment into positive or negative. ",
            "CR": "In this task, you're given a customer review of a product sold online, and you need to classify its sentiment into positive or negative. ",
            # Add more task mappings as needed
        }
        task_prefix = task_mapping.get(task, "")
    
    # Initial prompt
    messages = [
        {"role": "user", "content": f"{task_prefix}The essence of a sentence is often captured by its main subjects and actions. Provide the most accurate one-word meaning for this sentence: \"{input_text}\""}
    ]
    
    # Diversity follow-up prompts
    diversity_prompts = [
        f"That's one perspective. From another angle, what would be a different one-word meaning for this sentence: \"{input_text}\"",
        f"Give me a completely different perspective. What's another one-word meaning for: \"{input_text}\"",
        f"Consider the emotions conveyed. What's the one-word meaning focusing on sentiment for: \"{input_text}\"",
        f"Focus on the actions or events. What's the one-word meaning capturing the main action in: \"{input_text}\"",
        f"Think about the underlying concepts. What's the one-word meaning expressing the core concept in: \"{input_text}\"",
        f"Consider the entities involved. What's the one-word meaning highlighting the main entity in: \"{input_text}\""
    ]
    
    # Add dummy assistant responses and follow-up prompts
    for i in range(min(num_diverse - 1, len(diversity_prompts))):
        # Add a placeholder response from assistant (will be replaced during generation)
        messages.append({"role": "assistant", "content": "[ONE_WORD_MEANING]"})
        # Add the follow-up prompt from user
        messages.append({"role": "user", "content": diversity_prompts[i]})
    
    return messages





















#! Following are few shot prompts.



### PPFS0
pos0_template = (
#Summary
"""Summarize the input sentence in seven words.
 The input sentence is - {}
 What is your generated sentence?""",
" The output sentence is:")
pos0_template_examples=[]


pos0_template = [
    {"role": "user", "content": "Summarize the input sentence in a few words. Do not output any additional explanation. Only output the summary."},
    {"role": "assistant", "content": "Alright."},
    {"role": "user", "content": "*u6*"},  
]



pos1_template = [
    {"role": "user", "content": "Paraphrase the input sentence or phrase, providing an alternative expression with the same meaning. Please do not provide any alternative or reasoning or explanation."},
    {"role": "assistant", "content": "Alright."},
    {"role": "user", "content": "*u1*"},
    {"role": "assistant", "content": "*a1*"},
    {"role": "user", "content": "*u2*"},
    {"role": "assistant", "content": "*a2*"},
    {"role": "user", "content": "*u3*"},
    {"role": "assistant", "content": "*a3*"},
    {"role": "user", "content": "*u4*"},
    {"role": "assistant", "content": "*a4*"},
    {"role": "user", "content": "*u5*"},
    {"role": "assistant", "content": "*a5*"},
    {"role": "user", "content": "*u6*"},
    
]


pos1_template_examples=[
("A woman and the kid are walking along with the dog.", "A woman, a child, and a dog are walking together."),
("Girl holding a box of crayons, with a notebook in front of her.", "A girl is holding a box of crayons and there is a notebook in front of her."),
("The man is pedaling the tiny bike down the steps.", "The man is riding a small bike down the stairs."),
("A group of people cleaning a beach.", "Several individuals are cleaning the beach."),
("Most of them seemed to be dead or unconscious.", "Majority of them appeared lifeless or unconsciousness."),
("An older gentleman wearing a dark coat and hat leaning over a green object.", "A man, who appears to be elderly, is leaning over a green object while dressed in a dark coat and hat."),
("a man wearing white cleans dirt from the ground with a hose.", "A man in white clothing uses a hose to clean dirt off the ground."),
("An older woman with a nose ring, silver necklace, and decorative scarf is quietly sitting.", "A woman with a nose ring, silver necklace, and decorative scarf is sitting quietly."),
("Don't say it!", "Do not utter it!"),
("Woman making sushi from home.", "A woman is creating sushi in her own home."),
]



pos2_template = [
    {"role": "user", "content": "Rewrite the input sentence or phrase using different sentence structure and different words while preserving its original meaning. Please do not provide any alternative or reasoning or explanation."},
    {"role": "assistant", "content": "Alright."},
    {"role": "user", "content": "*u1*"},
    {"role": "assistant", "content": "*a1*"},
    {"role": "user", "content": "*u2*"},
    {"role": "assistant", "content": "*a2*"},
    {"role": "user", "content": "*u3*"},
    {"role": "assistant", "content": "*a3*"},
    {"role": "user", "content": "*u4*"},
    {"role": "assistant", "content": "*a4*"},
    {"role": "user", "content": "*u5*"},
    {"role": "assistant", "content": "*a5*"},
    {"role": "user", "content": "*u6*"},
        
]




pos2_template_examples=[
("A person is hanging upside down from power lines.", "Someone is hanging from power lines, but upside down."),
("a woman points with her left hand wearing navy blue clothes as a man wearing a visor backwards stands looking at her.", "A man wearing a visor backwards is standing and looking at a woman wearing navy blue clothes who is pointing with her left hand."),
("A young child is crying in the hands of an elderly male.", "An elderly male is holding a young child who is crying."),
("A little girl playing in a water jet fountain, with multiple children playing in the background.", "In the background, multiple children are playing while a little girl is playing in a water jet fountain."),
("A motorcycle rider leans in during a turn", "During a turn, the motorcycle rider leans in."),
("The data reliability process begins with two relatively simple steps.", "Two simple steps start data reliability process."),
("A woman is walking a blue bike across a road.", "Across the street, a lady is pushing a blue bicycle."),
("An older woman is sitting outside drawing with a brick building behind her.", "A brick building can be seen behind an older woman who is sitting outside and drawing."),
("You're right, said Tommy slowly.", "Tommy said slowly that you were right."),
("People are walking through an outdoor market.", "An outdoor market is bustling with people walking through it."),
]




pos3_template = [
    {"role": "user", "content": "Create a sentence or phrase that is also true, assuming the provided input sentence or phrase is true. Please do not provide any alternative or reasoning or explanation."},
    {"role": "assistant", "content": "Alright."},
    {"role": "user", "content": "*u1*"},
    {"role": "assistant", "content": "*a1*"},
    {"role": "user", "content": "*u2*"},
    {"role": "assistant", "content": "*a2*"},
    {"role": "user", "content": "*u3*"},
    {"role": "assistant", "content": "*a3*"},
    {"role": "user", "content": "*u4*"},
    {"role": "assistant", "content": "*a4*"},
    {"role": "user", "content": "*u5*"},
    {"role": "assistant", "content": "*a5*"},
    {"role": "user", "content": "*u6*"},
        
]

pos3_template_examples=[
("An older gentleman wearing a dark coat and hat leaning over a green object.", "An older man observing an object."),
("a man wearing white cleans dirt from the ground with a hose.", "A man is clearing dirt with a hose."),
("One baseball player is on the ground with his mouth open while another jumps above him.", "A baseball player with mouth agape is under another who is jumping."),
("Several people are posing for a photo with the naked cowboy in NYC.", "People are taking pictures with an unclothed individual."),
("A middle-aged woman is getting her hair done in a barber shop with polka-dotted walls.", "She was having her hair done."),
("A brown dog running on the beach near the ocean.", "The dog is running outside."),
("Woman making sushi from home.", "There were some people making homemade sushi."),
("An Asian skateboarder in a black shirt and fitted jeans shows off a trick.", "A skateboarder in black shows off a trick."),
("A woman and the kid are walking along with the dog.","The people are walking."),
("A shirtless child plays with an adult indoors.","The people are inside.")
]





pos4_template = [
    {"role": "user", "content": "Provide a concise paraphrase of the input sentence or phrase, maintaining the core meaning while altering the words and sentence structure. Feel free to omit some of the non-essential details like adjectives or adverbs. Please do not provide any alternative or reasoning or explanation."},
    {"role": "assistant", "content": "Alright."},
    {"role": "user", "content": "*u1*"},
    {"role": "assistant", "content": "*a1*"},
    {"role": "user", "content": "*u2*"},
    {"role": "assistant", "content": "*a2*"},
    {"role": "user", "content": "*u3*"},
    {"role": "assistant", "content": "*a3*"},
    {"role": "user", "content": "*u4*"},
    {"role": "assistant", "content": "*a4*"},
    {"role": "user", "content": "*u5*"},
    {"role": "assistant", "content": "*a5*"},
    {"role": "user", "content": "*u6*"},
        
]


pos4_template_examples=[
("The data reliability process begins with two relatively simple steps.", "Two simple steps start data reliability process."),
("Bald white men stands in a shop with many shirts.", "A white bald man is in a shirt store."),
("People hang upside down as a roller coaster executes a spiral loop-the-loop.", "People are on a roller coaster."),
("I heard nothing. Mrs. Vandemeyer gazed round her fearfully.", "Mrs. Vandemeyer looked around fearfully as I heard nothing."),
("Dave paid no attention to where his feet were leading him, only vaguely aware that he was heading down a gully below the current construction job.", "Dave walked absentmindedly down a gully."),
("and that\"s been uh uh a problem you know to to the merchant people that that fish and stuff up here they run into that ice stuff in the winter and it breaks away", "The ice breaks away in the winter when merchants fish."),
("These changes provide EPA with approximate targets so that each of the scenarios can be mapped into the AMIGA model.", "EPA has created approximate targets to map each scenario into the AMIGA model."),
("An audience of senior citizens is sitting for some kind of presentation.", "A presentation is being given to an audience consisting of senior citizens."),
("A group of people are discussing what they need to buy from Walmart", "A Walmart shopping list is being discussed by a group of people."),
("A woman is on her hands and knees cleaning a rug.", "A lady is scrubbing a rug on her knees."),
]





#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################

def get_thinker_reasoning_prompt(sentence, task):
    """
    Generate a prompt for the Thinker LLM to determine the best transformation strategy.
    This function creates a prompt that asks the model to analyze the input sentence
    and decide on the optimal transformation strategy for improving its embeddings.
    
    Args:
        sentence (str): The input sentence to be transformed
        task (str): The task context (e.g., 'STS', 'Classification')
        
    Returns:
        list: A list of message dictionaries for the chat template
    """
    
    base_text = f"""You are an expert in natural language understanding and semantic representation. Your task is to help improve sentence embeddings by determining the BEST transformation strategy for the given sentence.

CONTEXT:
Sentence embeddings need to capture the semantic meaning of text beyond surface-level features. The goal is to transform the input sentence into a new sentence that preserves or enhances its meaning for embedding models.

Input sentence: "{sentence}"

Analyze this sentence carefully. Consider:
1. The key semantic concepts and relationships
2. Any ambiguities, vagueness, or missing information
3. The complexity level and density of information
4. The context or domain it belongs to

Based on your analysis, choose ONE of the following transformation strategies that would BEST improve the sentence's embeddings:

- Elaboration: Add explanatory details that clarify concepts.
- Simplification: Reduce complexity while preserving core meaning.
- Specificity: Replace vague terms with more precise language.
- Contextual framing: Add domain context to disambiguate meaning.
- Abstraction: Extract higher-level concepts from specific details.
- Paraphrasing: Restate using different but semantically equivalent phrasing.

Provide your reasoning for why the chosen strategy is optimal, considering:
- How it addresses weaknesses in the original sentence
- How it preserves or enhances the semantic meaning
- How it improves representation in the embedding space

End your response with "Strategy: [strategy name]" on a separate line.

EXAMPLES:

Input: "The cat sat on the mat."
Analysis: This is a simple, concrete sentence with clear subjects and actions. However, it lacks context about what kind of cat or why it's sitting there, which creates ambiguity in the embedding space. Adding more context would help distinguish this from other similar sentences.
Strategy: Contextual framing

Input: "The algorithm achieved state-of-the-art performance."
Analysis: This sentence is vague and lacks specific details about what algorithm, what task, and what metrics define "state-of-the-art." The embedding would be improved by adding these specifics.
Strategy: Specificity

Input: "Mitochondria generate ATP through cellular respiration."
Analysis: While technically accurate, this sentence assumes domain knowledge about biology. For general embedding purposes, elaborating on what mitochondria and ATP are would create a more self-contained semantic unit.
Strategy: Elaboration
"""

    # Add task-specific context if available
    task_contexts = {
        'STS': "\nTASK CONTEXT: For Semantic Textual Similarity (STS), the transformation should ensure the core meaning is preserved while adding semantic richness that helps the model better capture nuanced similarities between sentences.",
        'Retrieval': "\nTASK CONTEXT: For Retrieval tasks, the transformation should enhance distinctions between similar but different concepts, making the sentence more retrievable for relevant queries.",
        'Classification': "\nTASK CONTEXT: For Classification tasks, the transformation should emphasize category-defining features while maintaining the sentence's original class membership.",
        'Reranking': "\nTASK CONTEXT: For Reranking tasks, the transformation should preserve query relevance while enhancing the aspects that make the sentence more relevant or irrelevant.",
        'Clustering': "\nTASK CONTEXT: For Clustering tasks, the transformation should emphasize features that would help group this sentence with semantically similar sentences.",
        'PairClassification': "\nTASK CONTEXT: For Pair Classification tasks, the transformation should preserve or enhance the relationship (or lack thereof) between this sentence and potential pairs.",
        'Summarization': "\nTASK CONTEXT: For Summarization tasks, the transformation should preserve the key information while potentially making the sentence more concise or information-dense."
    }
    
    if task in task_contexts:
        base_text += task_contexts[task]
    
    # Return a properly formatted list of message dictionaries
    return [{"role": "user", "content": base_text}]

def get_transformation_debug_prompt(sentence, strategy, reasoning):
    """
    Generate a prompt for transforming a sentence based on a specific strategy.
    This includes debug information for the pipeline.
    
    Args:
        sentence (str): The input sentence to be transformed
        strategy (str): The chosen transformation strategy
        reasoning (str): The reasoning behind the strategy choice
        
    Returns:
        list: A list of message dictionaries for the chat template
    """
    
    strategy_examples = {
        "Elaboration": [
            ("The economy is improving.", "The economy is showing signs of improvement with unemployment rates dropping by 2% and consumer spending increasing across multiple sectors."),
            ("Renewable energy is important.", "Renewable energy sources such as solar, wind, and hydroelectric power are important for reducing carbon emissions and combating climate change."),
            ("The medication has side effects.", "The medication may cause several side effects, including drowsiness, dry mouth, and in rare cases, allergic reactions characterized by rash and difficulty breathing.")
        ],
        "Simplification": [
            ("The intricate network of neurons facilitates the transmission of electrochemical signals that enable cognitive processes.", "The brain's network of cells helps send signals that let us think."),
            ("The policy implementation necessitates comprehensive stakeholder engagement to mitigate potential adverse ramifications.", "The policy needs input from everyone involved to avoid problems."),
            ("The quantum mechanical description of atomic structures elucidates electronic configuration and bonding properties.", "Quantum mechanics explains how atoms are structured and how they bond together.")
        ],
        "Specificity": [
            ("The presentation was good.", "The presentation on renewable energy technologies featured clear data visualizations and compelling case studies from successful implementations in Scandinavia."),
            ("The animal moved quickly.", "The cheetah sprinted at 70 miles per hour across the savanna to catch the gazelle."),
            ("The device has several features.", "The iPhone 13 Pro offers a ProMotion display with 120Hz refresh rate, a triple-camera system with macro photography, and A15 Bionic chip with 5-core GPU.")
        ],
        "Contextual framing": [
            ("The model achieved 95% accuracy.", "In the context of medical imaging diagnostics, the deep learning model achieved 95% accuracy on detecting early-stage lung cancer from CT scans."),
            ("The reaction failed.", "In the context of organic synthesis using palladium catalysts, the reaction failed due to oxygen contamination in the inert atmosphere chamber."),
            ("The system scales effectively.", "In the context of distributed computing for financial services, the system scales effectively to handle peak trading volumes exceeding 10 million transactions per second.")
        ],
        "Abstraction": [
            ("The company laid off 500 workers at its Detroit plant.", "The manufacturing sector continues to experience workforce reductions as companies respond to economic pressures."),
            ("The patient experienced headache, nausea, and sensitivity to light.", "The patient displayed symptoms consistent with neurological disorders affecting sensory processing."),
            ("The website added a login button, remember me checkbox, and password reset link.", "The website implemented enhanced user authentication features to improve security and user experience.")
        ],
        "Paraphrasing": [
            ("The research indicates a strong correlation between exercise and improved mental health.", "Studies suggest there's a significant relationship between physical activity and better psychological wellbeing."),
            ("The company's revenue exceeded expectations in the third quarter.", "In Q3, the firm's income surpassed the anticipated financial projections."),
            ("The committee rejected the proposal due to budgetary constraints.", "Citing financial limitations, the board declined to approve the suggested initiative.")
        ]
    }
    
    # Get examples for the selected strategy
    examples = strategy_examples.get(strategy, strategy_examples["Paraphrasing"])
    examples_text = "\n".join([f"Original: {ex[0]}\nTransformed: {ex[1]}" for ex in examples])
    
    prompt = f"""### TRANSFORMATION DEBUG INFO ###
Original sentence: "{sentence}"
Selected strategy: {strategy}
Reasoning: {reasoning[:200]}...

Your task is to transform the original sentence using the {strategy} strategy. This transformation should preserve or enhance the semantic meaning while optimizing it for embedding models.

Here are examples of the {strategy} strategy:
{examples_text}

Create ONE transformation that applies the {strategy} strategy to the original sentence. Focus on creating a high-quality transformation that will produce better embeddings.

Transformed sentence:"""
    
    # Return a properly formatted list of message dictionaries
    return [{"role": "user", "content": prompt}]

def get_contrastive_transformations_prompt(sentence, task=None):
    """
    Generate a prompt for creating multiple semantic transformations of a sentence for the ContrastiveEOL approach.
    This approach focuses on generating three different types of transformations that maintain the core meaning
    while providing different perspectives.
    
    Args:
        sentence (str): The input sentence to transform
        task (str, optional): The task context for transformation
        
    Returns:
        list: A list of message dictionaries for the chat template
    """
    
    # Define the base prompt that instructs the model to generate the three transformation types
    base_prompt = f"""Transform the following input sentence into three different versions that preserve its core meaning but approach it from different angles. For each transformation, follow the specific instructions:

Input sentence: "{sentence}"

1. SEMANTIC CORE: Extract and express only the most essential meaning using different words. Be concise and direct.

2. PERSPECTIVE SHIFT: Rewrite the sentence from a different grammatical or conceptual perspective, such as changing active to passive voice or shifting the focus to a different entity in the sentence.

3. IMPLICATED INFORMATION: Include information that is implied but not explicitly stated in the original, while staying true to the likely intended meaning.

Each transformation should be concise, grammatically correct, and maintain the core meaning. Avoid adding unnecessary details or excessive wording.

Format your response as:
SEMANTIC CORE: [your transformed sentence]
PERSPECTIVE SHIFT: [your transformed sentence]
IMPLICATED INFORMATION: [your transformed sentence]

Do not include any additional explanations or commentary."""

    # Add task-specific context if provided
    task_contexts = {
        'STS': "Since this is for a Semantic Textual Similarity task, ensure all transformations preserve the exact meaning but with different phrasings.",
        'Retrieval': "Since this is for a Retrieval task, ensure the transformations preserve key retrievable concepts and entities.",
        'Classification': "Since this is for a Classification task, ensure the transformations maintain any features that would determine classification outcomes.",
        'Reranking': "Since this is for a Reranking task, focus on preserving the aspects that make this sentence relevant to potential queries.",
        'Clustering': "Since this is for a Clustering task, ensure the transformations preserve the thematic elements that would determine cluster membership.",
        'PairClassification': "Since this is for a Pair Classification task, preserve any relationship features that would be important when paired with other sentences.",
        'Summarization': "Since this is for a Summarization task, ensure the transformations prioritize the most important information."
    }
    
    if task in task_contexts:
        base_prompt = f"{base_prompt}\n\nINSTRUCTION: {task_contexts[task]}"
    
    return [{"role": "user", "content": base_prompt}]


def process_contrastive_outputs(output_text):
    """
    Process the output from the contrastive transformations prompt to extract the three transformations.
    
    Args:
        output_text (str): The raw output from the LLM
        
    Returns:
        dict: A dictionary with keys 'semantic_core', 'perspective_shift', and 'implicated_information',
              each containing the corresponding transformation
    """
    # Initialize with empty strings in case parsing fails
    transformations = {
        'semantic_core': '',
        'perspective_shift': '',
        'implicated_information': ''
    }
    
    # Define patterns to look for
    patterns = {
        'semantic_core': ['SEMANTIC CORE:', '1.', 'Semantic Core:'],
        'perspective_shift': ['PERSPECTIVE SHIFT:', '2.', 'Perspective Shift:'],
        'implicated_information': ['IMPLICATED INFORMATION:', '3.', 'Implicated Information:']
    }
    
    # Process output line by line
    lines = output_text.strip().split('\n')
    current_key = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line is a header
        found_header = False
        for key, headers in patterns.items():
            for header in headers:
                if line.startswith(header):
                    current_key = key
                    # Extract text after the header on the same line
                    after_header = line[len(header):].strip()
                    if after_header:
                        transformations[current_key] = after_header
                    found_header = True
                    break
            if found_header:
                break
                
        # If not a header and we have a current key, append to that transformation
        if not found_header and current_key and not transformations[current_key]:
            transformations[current_key] = line
    
    # If any transformation is still empty, use some fallback logic
    for key in transformations:
        if not transformations[key]:
            # Try a different parsing approach looking for the key phrase anywhere in a line
            for line in lines:
                for header in patterns[key]:
                    if header in line:
                        parts = line.split(header, 1)
                        if len(parts) > 1:
                            transformations[key] = parts[1].strip()
                            break
                if transformations[key]:
                    break
    
    return transformations


def get_diverse_transformations_prompt(sentence, task=None):
    """
    Generate a prompt for creating five diverse semantic transformations of a sentence
    for the DiverseGenEOL approach. This approach focuses on generating transformations
    that maximize semantic diversity.
    
    Args:
        sentence (str): The input sentence to transform
        task (str, optional): The task context for transformation
        
    Returns:
        list: A list of message dictionaries for the chat template
    """
    
    # Define the base prompt that instructs the model to generate the five transformation types
    base_prompt = f"""Transform the following input sentence into five different versions that preserve its core meaning but explore different semantic dimensions. For each transformation, follow the specific instructions:

Input sentence: "{sentence}"

1. CORE REPRESENTATION: Express only the most essential meaning in the most concise way possible.

2. ENTITY SUBSTITUTION: Replace key entities (nouns, verbs) with semantically equivalent alternatives. Use synonyms or related concepts.

3. CONCEPTUAL ABSTRACTION: Express the meaning at a higher conceptual level, focusing on the underlying idea rather than specific details.

4. SPECIFICITY ENHANCEMENT: Add precise, relevant details that clarify the meaning without changing it.

5. RELATIONAL REFRAMING: Restructure the sentence to highlight different relationships between the entities mentioned.

Each transformation should be concise, grammatically correct, and maintain the core meaning. Avoid verbosity, unnecessary details, and semantic drift.

Format your response as:
CORE REPRESENTATION: [your transformed sentence]
ENTITY SUBSTITUTION: [your transformed sentence]
CONCEPTUAL ABSTRACTION: [your transformed sentence]
SPECIFICITY ENHANCEMENT: [your transformed sentence]
RELATIONAL REFRAMING: [your transformed sentence]

Do not include any additional explanations or commentary."""

    # Add task-specific context if provided
    task_contexts = {
        'STS': "Since this is for a Semantic Textual Similarity task, ensure all transformations preserve the exact meaning but explore different ways of expressing it.",
        'Retrieval': "Since this is for a Retrieval task, ensure the transformations preserve key retrievable concepts while exploring different ways to express them.",
        'Classification': "Since this is for a Classification task, ensure the transformations maintain features that determine classification outcomes.",
        'Reranking': "Since this is for a Reranking task, focus on preserving aspects that determine relevance while exploring different semantic dimensions.",
        'Clustering': "Since this is for a Clustering task, ensure the transformations preserve thematic elements while exploring semantic diversity.",
        'PairClassification': "Since this is for a Pair Classification task, preserve relationship features while exploring different semantic perspectives.",
        'Summarization': "Since this is for a Summarization task, ensure the transformations prioritize important information while exploring different ways to express it."
    }
    
    if task in task_contexts:
        base_prompt = f"{base_prompt}\n\nINSTRUCTION: {task_contexts[task]}"
    
    return [{"role": "user", "content": base_prompt}]


def process_diverse_outputs(output_text):
    """
    Process the output from the diverse transformations prompt to extract the five transformations.
    
    Args:
        output_text (str): The raw output from the LLM
        
    Returns:
        dict: A dictionary with keys for each transformation type,
              each containing the corresponding transformation
    """
    # Initialize with empty strings in case parsing fails
    transformations = {
        'core_representation': '',
        'entity_substitution': '',
        'conceptual_abstraction': '',
        'specificity_enhancement': '',
        'relational_reframing': ''
    }
    
    # Define patterns to look for
    patterns = {
        'core_representation': ['CORE REPRESENTATION:', '1.', 'Core Representation:'],
        'entity_substitution': ['ENTITY SUBSTITUTION:', '2.', 'Entity Substitution:'],
        'conceptual_abstraction': ['CONCEPTUAL ABSTRACTION:', '3.', 'Conceptual Abstraction:'],
        'specificity_enhancement': ['SPECIFICITY ENHANCEMENT:', '4.', 'Specificity Enhancement:'],
        'relational_reframing': ['RELATIONAL REFRAMING:', '5.', 'Relational Reframing:']
    }
    
    # Process output line by line
    lines = output_text.strip().split('\n')
    current_key = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line is a header
        found_header = False
        for key, headers in patterns.items():
            for header in headers:
                if line.startswith(header):
                    current_key = key
                    # Extract text after the header on the same line
                    after_header = line[len(header):].strip()
                    if after_header:
                        transformations[current_key] = after_header
                    found_header = True
                    break
            if found_header:
                break
                
        # If not a header and we have a current key, append to that transformation
        if not found_header and current_key and not transformations[current_key]:
            transformations[current_key] = line
    
    # If any transformation is still empty, use some fallback logic
    for key in transformations:
        if not transformations[key]:
            # Try a different parsing approach looking for the key phrase anywhere in a line
            for line in lines:
                for header in patterns[key]:
                    if header in line:
                        parts = line.split(header, 1)
                        if len(parts) > 1:
                            transformations[key] = parts[1].strip()
                            break
                if transformations[key]:
                    break
    
    return transformations

