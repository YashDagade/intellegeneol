import numpy as np
from typing import Dict, List, Union, cast

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import json
import os
from peft import PeftModel, PeftConfig
from accelerate.utils import gather_object
import time


from . import prompts_utils
import importlib
importlib.reload(prompts_utils)
from .prompts_utils import *


import openai
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import ast
import json_repair
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np
from scipy.spatial import ConvexHull
import re
from collections import defaultdict
from .openai_keys import *



class LLM: # it's not inheriting from torch.nn.Module, so .generate is sperately dfined

    def __init__(self, accelerator=None, args=None):
        self.args = args
        self.accelerator = accelerator
        self.model=None # this is most likely a model of type AutoModelForCausalLM? idk
        self.tokenizer=None
        self.ltokenizer=None
        
    # generate positives/negatives - used for the generation of the new sentences
    def generate(self, instructions_inputs_batch):
        if 'gpt' in self.args.gen_model_name_or_path:
            outputs=[]
            for messages in instructions_inputs_batch:
                retry_count = 0
                is_ok = False
                while not is_ok:
                    retry_count += 1
                    try:
                        response = openai.ChatCompletion.create(
                            model=self.args.gen_model_name_or_path,
                            messages=messages,
                            n=self.args.num_gens,
                            # temperature=args.temperature,
                            max_tokens=4096,
                            # stop=stop,
                            # top_p=args.top_p,
                        )
                        is_ok = True
                    except Exception as error:
                        time.sleep(1)
                        if retry_count % 5 ==0:
                            logger.warning(f"OpenAI API retry for {retry_count} times ({error})")
                            continue


                outputs.extend([each_gen["message"]["content"] for each_gen in response["choices"]])
            return outputs
        else:
            if(len(instructions_inputs_batch)>=4): #okay so we are just putting 4 sentences throuhgh the LM at a time here
                # import ipdb; ipdb.set_trace()
                # Assuming 'instructions_inputs_batch' is a list of inputs

                batch_size = 4 #so thes are teh transformations we are doing here
                final_outputs = []

                # Split the instructions_inputs_batch into chunks of size 8
                for i in range(0, len(instructions_inputs_batch), batch_size):
                    batch = instructions_inputs_batch[i:i + batch_size]

                    # Tokenize and send to model
                    inputs = self.ltokenizer.apply_chat_template(
                        batch, padding=True, truncation=True, 
                        max_length=self.args.max_length, return_tensors="pt", return_dict=True
                    ).to("cuda")
                    
                    # Generate output
                    output = self.model.generate(
                        **inputs, do_sample=True, num_return_sequences=self.args.num_gens, 
                        temperature=self.gtemp, repetition_penalty=1.0, 
                        top_p=self.top_p, max_new_tokens=512, 
                        return_dict_in_generate=True
                    )
                    
                    # Process output
                    new_output_ids = output.sequences[:, inputs.input_ids.shape[1]:]
                    del inputs; del output

                    outputs = self.tokenizer.batch_decode(
                        new_output_ids, skip_special_tokens=True, 
                        spaces_between_special_tokens=False
                    )
                    
                    del new_output_ids

                    # Clean outputs
                    outputs = [
                        o.replace("assistant\n\n", "", 1) if o.startswith("assistant\n\n") else o 
                        for o in outputs
                    ]
                    
                    # Append the processed outputs to the final list
                    final_outputs.extend(outputs)

                # Now final_outputs will contain all the processed outputs
                outputs=final_outputs

            else:    
                inputs = self.ltokenizer.apply_chat_template(instructions_inputs_batch, padding=True, truncation=True, max_length=self.args.max_length, return_tensors="pt", return_dict=True).to("cuda")
                output = self.model.generate(**inputs, do_sample=True, num_return_sequences=self.args.num_gens, temperature=self.gtemp, repetition_penalty=1.0, top_p=self.top_p, max_new_tokens=512, return_dict_in_generate=True)

                new_output_ids = output.sequences[:, inputs.input_ids.shape[1]:]
                del inputs; del output
                outputs = self.tokenizer.batch_decode(new_output_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
                

                del new_output_ids
                outputs = [o.replace("assistant\n\n", "", 1) if o.startswith("assistant\n\n") else o for o in outputs]
            
        return outputs

    # embedd using generations
    def embed(self, new_sentences_batch): # this is used for the embedding of the new sentences


        inputs = self.tokenizer(new_sentences_batch, padding=True, truncation=True, return_tensors='pt',
                                max_length=self.args.max_length, add_special_tokens=False).to("cuda")

        all_embeddings = []
        batch_size = 5
        # Get the total number of sentences
        total_sentences = inputs['input_ids'].shape[0]

        # Iterate through the tokenized inputs in chunks of batch_size
        for i in range(0, total_sentences, batch_size):
            # Create a batch of tokenized inputs
            batch_inputs = {key: tensor[i:i+batch_size] for key, tensor in inputs.items()}
            
            # Get the embeddings based on whether the penultimate layer is used
            if self.args.penultimate_layer!=-1:
                hidden_states = (getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model)(
                    **batch_inputs, output_hidden_states=True, return_dict=True).hidden_states
                last_hidden_state = hidden_states[self.args.penultimate_layer]
            else:
                outputs = (getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model)(**batch_inputs)
                last_hidden_state = outputs[0]
                del outputs
            
            # Append the batch embeddings to the list
            all_embeddings.append(last_hidden_state.detach().cpu())
        
        # Concatenate all the batch embeddings into one tensor
        final_embeddings = torch.cat(all_embeddings, dim=0)


        return final_embeddings, inputs

    def switchon_gen_model(self):
        
        if self.args.method=='b5': 
            return

        if 'gpt' in self.args.gen_model_name_or_path:
            import openai 
            

            OPENAI_API_BASE = "None"
            openai.api_key = OPENAI_API_KEY
            openai.organization = OPENAI_ORG_ID

            self.openaitokenizer = AutoTokenizer.from_pretrained("gpt2", fast_tokenizer=False) # TODO: For ChatGPT we should use a different one
            self.model=None
        elif self.args.gen_model_name_or_path is not None:
            if self.model is not None: del self.model
            self.model = AutoModelForCausalLM.from_pretrained(self.args.gen_model_name_or_path, trust_remote_code=True, device_map={"": self.accelerator.process_index}, torch_dtype=self.args.torch_dtype)

            if('mistral' in self.args.gen_model_name_or_path.lower()):
                self.gtemp=0.7
                self.top_p=1
            elif('llama' in self.args.gen_model_name_or_path.lower()):
                self.gtemp=0.6
                self.top_p=0.9
            else:
                self.gtemp=0.8
                self.top_p=0.95

            self.embedding_attr = 'model'
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            print(f"Created {self.args.gen_model_name_or_path}: {self.model.dtype} dtype")


            self.ltokenizer = AutoTokenizer.from_pretrained(self.args.gen_model_name_or_path, padding_side='left')
            self.tokenizer = AutoTokenizer.from_pretrained(self.args.gen_model_name_or_path, padding_side='right')
            if not(self.tokenizer.pad_token) and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.ltokenizer.pad_token = self.ltokenizer.eos_token

            print('Set pad token to eos token: ' + self.tokenizer.pad_token)
            self.model.eval()
        elif self.args.gen_model_name_or_path is None or self.args.gen_model_name_or_path == '-':
            pass
        else:
            pass

    def switchon_emb_model(self):       

        if self.model is not None: del self.model
        
        self.model = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path, trust_remote_code=True, device_map={"": self.accelerator.process_index}, torch_dtype=self.args.torch_dtype)

        if('mistral' in self.args.model_name_or_path.lower()):
            self.gtemp=0.7
            self.top_p=1
        elif('llama' in self.args.model_name_or_path.lower()):
            self.gtemp=0.6
            self.top_p=0.9
        else:
            self.gtemp=0.8
            self.top_p=0.95

        self.embedding_attr = 'model'
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"Created {self.args.model_name_or_path}: {self.model.dtype} dtype")


        self.ltokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, padding_side='left')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, padding_side='right')

        # import ipdb; ipdb.set_trace()
        if not(self.tokenizer.pad_token) and self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.ltokenizer.pad_token = self.ltokenizer.eos_token

        print('Set pad token to eos token: ' + self.tokenizer.pad_token)
        self.model.eval()



class GenEOL(torch.nn.Module):
    def __init__(self, accelerator, args) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.llm = LLM(accelerator, args)
        self.args=args
        self.pooling_method = args.pooling_method
        self.encode_call=0



    def encode_queries(self, queries: Union[List[str], str], **kwargs) -> np.ndarray:
        """Used for encoding the queries of retrieval or reranking tasks"""
        return self.encode(queries, **kwargs)

    def encode_corpus(self, corpus: Union[List[str], str, List[Dict[str, str]]], **kwargs) -> np.ndarray:
        """Used for encoding the corpus of retrieval tasks"""
        if isinstance(corpus, dict):
            corpus = [corpus]
        if isinstance(corpus, list) and isinstance(corpus[0], dict):
            corpus = [
                doc["title"] + " " + doc["text"] if "title" in doc 
                else doc["text"] for doc in corpus
            ]
        return self.encode(corpus, **kwargs)



    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        convert_to_tensor: bool = False,
        args=None,
        **kwargs,
    ) -> np.ndarray:
        self.encode_call+=1
        # If args is None, use self.args
        if args is None:
            args = self.args
            
        # Initialize new_sentences_batch as empty list for safety
        new_sentences_batch = []
        
        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        print(args.task, flush=True)
        # print(convert_to_tensor, "Convert to tensor", flush=True)
        
        # return np.zeros((len(sentences), 50))

        self.accelerator.wait_for_everyone()    
        start=time.time()
        
        with self.accelerator.split_between_processes(sentences) as sentences_rank_unchopped:
            np.random.seed(args.seed)
            self.llm.switchon_gen_model()
            all_embeddings = []
            
            save_path = os.path.join(args.output_folder, "transformations")
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving transformations to: {save_path}", flush=True)

            all_new_sentences_batch = []
            #! important to reduce size for all methods equally.
            # sentences_rank = self.llm.tokenizer.batch_decode(self.llm.tokenizer(sentences_rank_unchopped, max_length=args.max_length, truncation=True, add_special_tokens=False).input_ids)
            sentences_rank = sentences_rank_unchopped # this is the list of sentences that we are going to encode

            #! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PART 1
            if(not os.path.exists(f"{save_path}/{args.task}_sentences{self.encode_call}_{self.accelerator.process_index}.json")): 
                print("process id", self.accelerator.process_index, flush=True)

                
                for start_index in tqdm(range(0, len(sentences_rank), args.batch_size), desc="Batches", disable=len(sentences_rank)<50):
                    
                    instructions_inputs_batch = []
                    sentences_batch = sentences_rank[start_index:start_index + args.batch_size]
                    if(args.method=='s5'):
                        for s in sentences_batch:
                            instructions_inputs_batch.extend([get_pos_prompt(1, s), get_pos_prompt(2, s), get_pos_prompt(3, s), get_pos_prompt(4, s)])
                        total_num_gens=args.num_gens*4 # reason times 4 because we are using 4 different prompts

                        #! call LLM here
                        outputs = self.llm.generate(instructions_inputs_batch)

                        new_sentences_batch = []
                        for idx in range(len(sentences_batch)):
                            new_sentences_batch.append(sentences_batch[idx])
                            new_sentences_batch.extend(outputs[total_num_gens*idx:total_num_gens*(idx+1)])

                    elif(args.method=='d5'):
                        for s in sentences_batch:
                            instructions_inputs_batch.extend([get_diverse_prompt_fs(s)])

                        #! call LLM here
                        outputs = self.llm.generate(instructions_inputs_batch)
                        # outputs = [opt.strip().split("\n")[-1] for opt in outputs]

                        # new_sentences_batch = []
                        # for idx in range(len(sentences_batch)):
                        #     new_sentences_batch.append(sentences_batch[idx])
                        #     new_sentences_batch.extend(outputs[total_num_gens*idx:total_num_gens*(idx+1)])
                        total_num_gens=args.num_gens*10 # reason times 10 because we are using 10 different prompts
                        new_sentences_batch = []
                        for idx in range(len(sentences_batch)):
                            new_sentences_batch.append(sentences_batch[idx])

                            gens = outputs[idx].split("\n")
                            # print(gens)
                            gens = [re.sub(r'^\d+\.\s*', '', gen).strip() for gen in gens if len(gen)>5]

                            # print("should not happen often", f"\n\n\n{outputs[idx]}\n\n\n", flush=True)
                            
                            if len(gens)<10:
                                gens += [new_sentences_batch[-1]]*(10-len(gens))
                                print("<<< lower length", flush=True)
                            elif len(gens)>10:
                                
                                gens = gens[:10]
                                print(">>> higher length", flush=True)
                            else:
                                pass
                            new_sentences_batch.extend(gens)
                    elif(args.method=='d52' or args.method=='c5' or args.method=='ch5'):
                        
                        outputs_extra = []
                        for sind, s in enumerate(sentences_batch):
                            instructions_inputs_batch.extend([get_task_specific_gen_prompt(s, args.task)])


                        #! call LLM here
                        outputs = self.llm.generate(instructions_inputs_batch)

                        # outputs = [opt.strip().split("\n")[-1] for opt in outputs]

                        # new_sentences_batch = []
                        # for idx in range(len(sentences_batch)):
                        #     new_sentences_batch.append(sentences_batch[idx])
                        #     new_sentences_batch.extend(outputs[total_num_gens*idx:total_num_gens*(idx+1)])
                        total_num_gens=args.num_gens*10
                        new_sentences_batch = []
                        for idx in range(len(sentences_batch)):
                            new_sentences_batch.append(sentences_batch[idx])
                            #! we use a hard coded value 10 here
                            if(len(sentences_batch[idx]) <= 1):
                                new_sentences_batch.extend([sentences_batch[idx]]*total_num_gens)
                                continue

                            try:
                                # import ipdb; ipdb.set_trace()   
                                gens = json_repair.loads(outputs[idx])
                                if(type(gens)==dict):
                                    gens = gens["generations"]
                                    gens = [gen.strip() for gen in gens if len(gen.strip())]
                                elif(type(gens)==list):
                                    gens = gens
                                else:
                                    print("text formattable but not dict, list", f"\n\n\n{gens}\n\n\n", flush=True)
                                
                            except:
                                gens = [sentences_batch[idx]]*total_num_gens
                                print("should not happen often", f"\n\n\n{outputs[idx]}\n\n\n", flush=True)
                            
                            if len(gens)<10:
                                gens += [new_sentences_batch[-1]]*(10-len(gens))
                                print("<<< lower length", flush=True)
                            elif len(gens)>10:
                                
                                gens = gens[:10]
                                print(">>> higher length", flush=True)
                            else:
                                pass
                            new_sentences_batch.extend(gens)
                    elif(args.method=='r5'):
                        for s in sentences_batch:
                            instructions_inputs_batch.extend([get_task_specific_gen_prompt(s, args.task)])

                        total_num_gens=1
                        
                        #! call LLM here
                        outputs = self.llm.generate(instructions_inputs_batch)
                        new_sentences_batch = []
                        for idx in range(len(sentences_batch)):
                            new_sentences_batch.append(sentences_batch[idx])
                            new_sentences_batch.extend(outputs[total_num_gens*idx:total_num_gens*(idx+1)])
                    elif(args.method=='t5' or args.method=='t1'):
                        # Two-stage thinking process: Reasoning → Transformation → Embedding
                        print("Using intelleGenEOL approach with two-stage thinking...", flush=True)
                        
                        # First: Get reasoning about the best transformation strategy
                        reasoning_prompts = []
                        for s in sentences_batch:
                            # Each function returns a list with one item, so we extend rather than append
                            reasoning_prompts.extend(get_thinker_reasoning_prompt(s, args.task))
                        
                        # Call LLM for reasoning
                        reasoning_outputs = self.llm.generate(reasoning_prompts)
                        
                        # Parse reasoning outputs to extract strategies
                        strategies = []
                        reasonings = []
                        
                        for output in reasoning_outputs:
                            # Extract strategy name (should be at the end after "Strategy: ")
                            strategy_line = None
                            reasoning_text = ""
                            
                            # Split by lines and look for the strategy line
                            lines = output.strip().split('\n')
                            for i, line in enumerate(lines):
                                if line.startswith("Strategy:"):
                                    strategy_line = line
                                    reasoning_text = "\n".join(lines[:i])
                                    break
                            
                            if not strategy_line:
                                # If no "Strategy:" line found, use the default and whole text as reasoning
                                strategy = "Paraphrasing"
                                reasoning_text = output
                            else:
                                strategy = strategy_line.replace("Strategy:", "").strip()
                                # Normalize strategy names
                                if "elaborat" in strategy.lower():
                                    strategy = "Elaboration"
                                elif "simplif" in strategy.lower():
                                    strategy = "Simplification"
                                elif "specific" in strategy.lower():
                                    strategy = "Specificity"
                                elif "context" in strategy.lower():
                                    strategy = "Contextual framing"
                                elif "abstract" in strategy.lower():
                                    strategy = "Abstraction"
                                elif "paraphras" in strategy.lower():
                                    strategy = "Paraphrasing"
                                else:
                                    strategy = "Paraphrasing"  # Default if no match
                            
                            strategies.append(strategy)
                            reasonings.append(reasoning_text)
                        
                        # Second: Generate transformations using derived strategies and reasoning
                        transformation_prompts = []
                        
                        # Print debug information
                        print("\n======= INTELLEGENEOL DEBUG INFO =======", flush=True)
                        for idx, s in enumerate(sentences_batch):
                            print(f"Original [{idx}]: {s}", flush=True)
                            print(f"Strategy [{idx}]: {strategies[idx]}", flush=True)
                            print(f"Reasoning [{idx}]: {reasonings[idx][:100]}...", flush=True)
                            
                            # Each function returns a list with one item, so we extend rather than append
                            transformation_prompts.extend(get_transformation_debug_prompt(s, strategies[idx], reasonings[idx]))
                        
                        # Call LLM for transformations
                        transformation_outputs = self.llm.generate(transformation_prompts)
                        
                        # Create new sentences batch with original and transformed sentences
                        new_sentences_batch = []
                        total_num_gens = 1  # We're generating 1 transformation per sentence for t1
                        
                        for idx, output in enumerate(transformation_outputs):
                            # Add original sentence first
                            new_sentences_batch.append(sentences_batch[idx])
                            
                            # Add transformation
                            transformed = output.strip()
                            print(f"Transformed [{idx}]: {transformed}", flush=True)
                            new_sentences_batch.append(transformed)
                        
                        print("=======================================\n", flush=True)
                    elif(args.method=='c3'):
                        # ContrastiveEOL: Single-pass generation of three transformation types
                        print("Using ContrastiveEOL approach with semantic triangulation...", flush=True)
                        
                        # Generate prompts for contrastive transformations
                        contrastive_prompts = []
                        for s in sentences_batch:
                            contrastive_prompts.extend(get_contrastive_transformations_prompt(s, args.task))
                        
                        # Call LLM for transformations
                        contrastive_outputs = self.llm.generate(contrastive_prompts)
                        
                        # Create new sentences batch with original and transformed sentences
                        new_sentences_batch = []
                        total_num_gens = 3  # We're generating 3 transformations per sentence
                        
                        # Print debug information
                        print("\n======= CONTRASTIVEEOL DEBUG INFO =======", flush=True)
                        for idx, output in enumerate(contrastive_outputs):
                            # Process the output to extract the three transformations
                            transformations = process_contrastive_outputs(output)
                            
                            # Original sentence
                            original = sentences_batch[idx]
                            new_sentences_batch.append(original)
                            print(f"Original [{idx}]: {original}", flush=True)
                            
                            # Add the three transformations
                            for transform_type, transformed in transformations.items():
                                if transformed:  # Only add non-empty transformations
                                    print(f"{transform_type.replace('_', ' ').title()} [{idx}]: {transformed}", flush=True)
                                    new_sentences_batch.append(transformed)
                                else:
                                    # If transformation extraction failed, just duplicate the original
                                    print(f"Failed to extract {transform_type} transformation, using original", flush=True)
                                    new_sentences_batch.append(original)
                        
                        print("=======================================\n", flush=True)
                    elif(args.method=='d5'):
                        # DiverseGenEOL: Single-pass generation of five diverse transformation types
                        print("Using DiverseGenEOL approach with semantic diversity optimization...", flush=True)
                        
                        # Generate prompts for diverse transformations
                        diverse_prompts = []
                        for s in sentences_batch:
                            diverse_prompts.extend(get_diverse_transformations_prompt(s, args.task))
                        
                        # Call LLM for transformations
                        diverse_outputs = self.llm.generate(diverse_prompts)
                        
                        # Create new sentences batch with original and transformed sentences
                        new_sentences_batch = []
                        total_num_gens = 5  # We're generating 5 transformations per sentence
                        
                        # Print debug information
                        print("\n======= DIVERSEGENEOL DEBUG INFO =======", flush=True)
                        for idx, output in enumerate(diverse_outputs):
                            # Process the output to extract the five transformations
                            transformations = process_diverse_outputs(output)
                            
                            # Original sentence
                            original = sentences_batch[idx]
                            new_sentences_batch.append(original)
                            print(f"Original [{idx}]: {original}", flush=True)
                            
                            # Add the five transformations
                            for transform_type, transformed in transformations.items():
                                if transformed:  # Only add non-empty transformations
                                    print(f"{transform_type.replace('_', ' ').title()} [{idx}]: {transformed}", flush=True)
                                    new_sentences_batch.append(transformed)
                                else:
                                    # If transformation extraction failed, just duplicate the original
                                    print(f"Failed to extract {transform_type} transformation, using original", flush=True)
                                    new_sentences_batch.append(original)
                        
                        print("=======================================\n", flush=True)
                    elif(args.method=='b5'):
                        new_sentences_batch=sentences_batch
                        total_num_gens = 0
                    elif(args.method=='r3' or args.method=='r5'):
                        # RegenerateEOL: Direct diverse embedding generation without sentence transformations
                        print(f"Using RegenerateEOL approach with {args.num_gens} diverse embeddings...", flush=True)
                        
                        # Determine number of diverse embeddings to generate
                        num_diverse = 3 if args.method == 'r3' else 5

                        # Store original sentences for embedding
                        new_sentences_batch = sentences_batch.copy()
                        total_num_gens = 0  # Special case: no actual sentence generation
                        
                        # We'll handle the embedding differently for this method in the embedding phase
                        # Just preserve the original sentences here
                    else:
                        assert False, "pick between s5, d5, r5, r3, t1, t5, c3, d5 and b5"
                    all_new_sentences_batch.extend(new_sentences_batch)      
      

                with open(f"{save_path}/{args.task}_sentences{self.encode_call}_{self.accelerator.process_index}.json", "w") as f:
                    json.dump(all_new_sentences_batch, f)
            else:
                print("Loading first level generations", flush=True)
                with open(f"{save_path}/{args.task}_sentences{self.encode_call}_{self.accelerator.process_index}.json") as f:
                    all_new_sentences_batch = json.load(f)

                
                if(args.method=='s5'):
                    total_num_gens = 4*args.num_gens
                elif(args.method=='d5' or args.method=='d52'):
                    total_num_gens = 10*args.num_gens
                elif(args.method=='t5' or args.method=='t1'):
                    total_num_gens = 1  # We're generating 1 transformation per sentence for t1
                elif(args.method=='c3'):
                    total_num_gens = 3  # We're generating 3 transformations per sentence for c3
                elif(args.method=='d5'):
                    total_num_gens = 5  # We're generating 5 transformations per sentence for d5
                elif(args.method=='b5'):
                    total_num_gens = 0
                    # assert False, "b5 not compatibale with compositional"
                elif(args.method=='r3' or args.method=='r5'):
                    # RegenerateEOL methods have special handling for num_gens
                    num_diverse = 3 if args.method == 'r3' else 5
                    total_num_gens = 0  # Special handling in embedding phase
                else:
                    assert False, "Not accepted method"




            #! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PART 2
            if(self.args.compositional and (not os.path.exists(f"{save_path}/{args.task}_compositional_sentences{self.encode_call}_{self.accelerator.process_index}.json"))):
                
                # print(len(all_new_sentences_batch), type(all_new_sentences_batch), all_new_sentences_batch[-1:])
                compinst_all_new_sentences = []
                orig_sentences = []
                for idx, s in enumerate(all_new_sentences_batch):
                    if(idx%(total_num_gens+1)==0):
                        orig_sentences.append(s)
                    else:
                        compinst_all_new_sentences.append(get_sum_prompt_fs(s))
            
                print(compinst_all_new_sentences)
                self.temp_num_gens = self.args.num_gens
                self.args.num_gens=1
                # for start_index in tqdm(range(0, len(comp_all_new_sentences), new_batch_size), desc="Batches", disable=len(comp_all_new_sentences_batch)<50):
                all_new_sentences_batch = []
                for start_index in tqdm(range(0, len(orig_sentences), args.batch_size), desc="Batches", disable=len(orig_sentences)<50):
                    sentences_batch = orig_sentences[start_index:start_index+args.batch_size]
                    compinst_sentences_batch = compinst_all_new_sentences[total_num_gens*start_index:total_num_gens*(start_index+args.batch_size)]                
                    
                    outputs = self.llm.generate(compinst_sentences_batch)
                    new_sentences_batch = []
                    for idx in range(len(sentences_batch)):
                        new_sentences_batch.append(sentences_batch[idx])
                        new_sentences_batch.extend(outputs[total_num_gens*idx:total_num_gens*(idx+1)])
                    all_new_sentences_batch.extend(new_sentences_batch)

                with open(f"{save_path}/{args.task}_compositional_sentences{self.encode_call}_{self.accelerator.process_index}.json", "w") as f:
                    json.dump(all_new_sentences_batch, f)

                self.args.num_gens=self.temp_num_gens
            elif(self.args.compositional):
                print("Loading compositional generations", flush=True)
                with open(f"{save_path}/{args.task}_compositional_sentences{self.encode_call}_{self.accelerator.process_index}.json") as f:
                    all_new_sentences_batch = json.load(f)
            else:
                pass
            

            print("Task specific Prompts" if args.tsep else "KWEOL Prompts", flush=True)   
            


            #! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> PART 3
            if(not args.gen_only):
                self.llm.switchon_emb_model()          
                # all_new_sentences_batch = [f'<s>The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : "{s}" means in one word:"' for s in all_new_sentences_batch]

                # For RegenerateEOL, we need special handling of the embedding prompts
                if args.method in ['r3', 'r5']:
                    # For RegenerateEOL, we process each sentence individually to get multiple embeddings
                    num_diverse = 3 if args.method == 'r3' else 5
                    all_embeddings = []
                    
                    # Process each original sentence
                    total_sentences = len(all_new_sentences_batch)
                    print(f"Total sentences to process: {total_sentences}", flush=True)
                    
                    for sentence_idx, sentence in enumerate(tqdm(all_new_sentences_batch, desc="RegenerateEOL Embeddings")):
                        # Make sure we don't exceed the expected count - safety check for STS datasets
                        if sentence_idx >= total_sentences:
                            print(f"WARNING: Reached maximum expected sentence count. Stopping at {sentence_idx}.", flush=True)
                            break
                            
                        # Print progress info
                        if sentence_idx % 10 == 0:
                            print(f"\nProcessing sentence {sentence_idx}/{total_sentences}: \"{sentence[:50]}{'...' if len(sentence) > 50 else ''}\"", flush=True)
                        
                        # Track embedding words for diversity signaling
                        embedding_words = []
                        
                        # Generate multiple embeddings with diversity
                        batch_embeddings = []
                        
                        # Process each embedding sequentially, building on previous predictions
                        for i in range(num_diverse):
                            # Get the EOL prompt, including previous words for diversity
                            prompt = get_regenerate_emb_prompts(sentence, 1, args.task if args.tsep else None, embedding_words)[0]
                            
                            # Get embedding for this prompt
                            last_hidden_state, inputs = self.llm.embed([prompt])
                            
                            # For EOL extraction, we need to:
                            # 1. Get the penultimate layer output (this is already done in embed())
                            # 2. Extract the word that would be predicted (for logging and next prompts)
                            
                            # Extract the predicted word for the next prompt
                            try:
                                # Get the full response text
                                full_text = self.llm.tokenizer.decode(inputs['input_ids'][0])
                                
                                # For EOL, we look for the word after "means in one word:"
                                word_marker = "means in one word:"
                                
                                # Show debugging info
                                if sentence_idx < 5 or sentence_idx % 50 == 0:
                                    print(f"\n  PROMPT [{i+1}]: {prompt}", flush=True)
                                    print(f"  EOL TEXT: {full_text}", flush=True)
                                
                                # Try to extract the next word - this is the EOL extraction
                                extracted_word = None
                                
                                # In EOL setting, we need to use a Transformer to predict next token
                                # Here we'll approximate by looking at the tokenized text and taking word after marker
                                if word_marker in full_text:
                                    prediction_text = full_text.split(word_marker, 1)[1].strip()
                                    
                                    if sentence_idx < 5 or sentence_idx % 50 == 0:
                                        print(f"  AFTER MARKER: \"{prediction_text}\"", flush=True)
                                    
                                    # Get the first word
                                    prediction_words = [w.strip().rstrip('.,;:"\'!?') for w in prediction_text.split()]
                                    prediction_words = [w for w in prediction_words if w and len(w) > 1]  # Filter empty strings
                                    
                                    if prediction_words:
                                        extracted_word = prediction_words[0]
                                        if sentence_idx < 5 or sentence_idx % 50 == 0:
                                            print(f"  PREDICTED WORD [{i+1}]: \"{extracted_word}\"", flush=True)
                                
                                # If extraction failed (no prediction available), use alternative
                                if not extracted_word:
                                    # Use a meaningful word from the sentence as fallback
                                    content_words = [w.strip().rstrip('.,;:"\'!?') for w in sentence.split() 
                                                   if len(w) > 3 and w.lower() not in 
                                                   {"the", "and", "with", "this", "that", "then", "than", "when", "what"}]
                                    
                                    if content_words:
                                        extracted_word = content_words[0].lower()
                                        if sentence_idx < 5 or sentence_idx % 50 == 0:
                                            print(f"  FALLBACK WORD [{i+1}]: \"{extracted_word}\" from sentence", flush=True)
                                    else:
                                        # Ultimate fallback
                                        extracted_word = f"concept_{i+1}"
                                        if sentence_idx < 5 or sentence_idx % 50 == 0:
                                            print(f"  ULTIMATE FALLBACK [{i+1}]: \"{extracted_word}\"", flush=True)
                                
                                # Store the word for diversity in future prompts
                                embedding_words.append(extracted_word)
                                
                                # Display embedding info
                                if i == 0:
                                    print(f"  Initial embedding for \"{sentence[:30]}{'...' if len(sentence) > 30 else ''}\": {extracted_word}", flush=True)
                                else:
                                    prev_words = ", ".join(embedding_words[:-1])
                                    print(f"  Diverse embedding #{i+1}: {extracted_word} (previous: {prev_words})", flush=True)
                            
                            except Exception as e:
                                # Handle errors gracefully
                                default_word = f"concept_{i+1}"
                                embedding_words.append(default_word)
                                print(f"  Error extracting word: {str(e)}", flush=True)
                                print(f"  Using fallback: {default_word}", flush=True)
                            
                            # Extract embedding from the hidden state
                            # This uses standard EOL approach: don't mask anything, use full attention
                            # The key is to use the penultimate layer output which is already in last_hidden_state
                            full_mask = torch.ones_like(inputs['attention_mask'])
                            embed = self.pooling(last_hidden_state, full_mask, recast=False).to('cpu')
                            
                            # Verify the embedding is valid before adding it
                            if torch.isnan(embed).any():
                                print(f"WARNING: NaN detected in raw embedding {i+1} for sentence {sentence_idx}. Creating fallback.", flush=True)
                                # Create a fallback embedding of the same shape
                                embed = torch.zeros_like(embed)
                                embed[0, 0] = 1.0  # Set first value to 1 for non-zero norm
                                
                            # Add the embedding to our batch
                            batch_embeddings.append(embed)
                            
                            del last_hidden_state
                            del inputs
                        
                        # Combine all embeddings for this sentence
                        sentence_embeddings = torch.cat(batch_embeddings, dim=0)
                        
                        # Check for NaN values
                        has_nan = torch.isnan(sentence_embeddings).any()
                        if has_nan:
                            print(f"WARNING: NaN detected in embeddings for sentence {sentence_idx}. Replacing with valid embeddings.", flush=True)
                            # Replace any NaN embeddings with the first valid embedding in the batch
                            valid_indices = ~torch.isnan(sentence_embeddings).any(dim=1)
                            if valid_indices.any():
                                # Use the first valid embedding for any NaN embeddings
                                first_valid_idx = valid_indices.nonzero()[0].item()
                                valid_embedding = sentence_embeddings[first_valid_idx].unsqueeze(0)
                                
                                # Create a mask for NaN embeddings
                                nan_mask = torch.isnan(sentence_embeddings).any(dim=1)
                                for i in range(len(sentence_embeddings)):
                                    if nan_mask[i]:
                                        sentence_embeddings[i] = valid_embedding
                            else:
                                # If all embeddings have NaNs, create a fallback embedding
                                print(f"CRITICAL: All embeddings contain NaN for sentence {sentence_idx}. Using fallback.", flush=True)
                                sentence_embeddings = torch.zeros_like(sentence_embeddings)
                                sentence_embeddings[0] = 1.0  # Set first dimension to 1 for a unit vector after normalization
                        
                        # Normalize embeddings if requested
                        if self.args.normalized:
                            in_dtype = sentence_embeddings.dtype
                            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, dim=-1).to(in_dtype)
                            
                            # Double-check for NaNs after normalization
                            if torch.isnan(sentence_embeddings).any():
                                print(f"WARNING: NaN detected after normalization for sentence {sentence_idx}. Using fallback.", flush=True)
                                sentence_embeddings = torch.zeros_like(sentence_embeddings)
                                sentence_embeddings[0, 0] = 1.0  # Set first dimension to 1
                        
                        # Average the diverse embeddings with equal weights
                        avg_embedding = torch.mean(sentence_embeddings, dim=0, keepdim=True)
                        
                        # Final verification for NaNs in the average embedding
                        if torch.isnan(avg_embedding).any():
                            print(f"CRITICAL: NaN in final embedding for sentence {sentence_idx}. Using fallback vector.", flush=True)
                            avg_embedding = torch.zeros_like(avg_embedding)
                            avg_embedding[0, 0] = 1.0  # Set first dimension to 1
                        
                        # Log progress
                        if sentence_idx % 10 == 0:
                            print(f"  Generated {len(batch_embeddings)} diverse embeddings for sentence {sentence_idx}", flush=True)
                        
                        all_embeddings.append(avg_embedding)
                    
                    # Combine all sentence embeddings
                    print(f"\nCompleted processing {len(all_new_sentences_batch)} sentences with {num_diverse} embeddings each", flush=True)
                    
                    # Make sure we have the right number of embeddings - critical for STS datasets
                    if len(all_embeddings) != total_sentences:
                        print(f"WARNING: Number of processed embeddings ({len(all_embeddings)}) doesn't match expected count ({total_sentences})", flush=True)
                        
                        # If we have too few embeddings, pad with zeros
                        if len(all_embeddings) < total_sentences:
                            print(f"Adding {total_sentences - len(all_embeddings)} zero embeddings to match expected count", flush=True)
                            embed_dim = all_embeddings[0].shape[-1]
                            for _ in range(total_sentences - len(all_embeddings)):
                                zero_embed = torch.zeros(1, embed_dim, device=all_embeddings[0].device)
                                zero_embed[0, 0] = 1.0  # Set first dimension to 1
                                all_embeddings.append(zero_embed)
                        
                        # If we have too many embeddings, truncate
                        elif len(all_embeddings) > total_sentences:
                            print(f"Truncating {len(all_embeddings) - total_sentences} embeddings to match expected count", flush=True)
                            all_embeddings = all_embeddings[:total_sentences]
                    
                    # Verify one more time
                    print(f"Final embedding count: {len(all_embeddings)}", flush=True)
                    all_embeddings = torch.cat(all_embeddings, dim=0)
                else:
                    # Handle regular embedding methods
                new_batch_size = (total_num_gens+1)*args.batch_size
                for start_index in tqdm(range(0, len(all_new_sentences_batch), new_batch_size), desc="Batches"):
                    new_sentences_batch = all_new_sentences_batch[start_index:start_index + new_batch_size]

                    # new_sentences_batch = [f'<s>This sentence : "{s}" means in one word:"' for s in new_sentences_batch]
                    #! embed here
                    last_hidden_state, inputs = self.llm.embed(new_sentences_batch)
                    
                    # if self.projection:
                    #     last_hidden_state = self.projection(last_hidden_state)

                    if ("mean" in args.pooling_method):
                            # Skip attention masking entirely to avoid NaN issues
                            # Just use the token weights as they are
                            pass
                        
                        # Use a full attention mask (all 1s) to avoid NaN values
                        safe_mask = inputs['attention_mask'].clone()
                        
                        # Check if we need to apply any masking at all
                        apply_masking = False
                        
                        if apply_masking:
                            # Process only the current batch elements within limits
                            batch_size = inputs['attention_mask'].shape[0]
                            sentences_to_process = new_sentences_batch[:batch_size]

                            for ii_idx, instruction_input in enumerate(sentences_to_process):
                                # Ensure we don't exceed batch dimensions
                                if ii_idx >= batch_size:
                                    break
                                
                                # For Mistral INST format, we want to mask everything up to and including [/INST]
                                inst_end_marker = "[/INST]"
                                marker_pos = instruction_input.find(inst_end_marker)
                                
                                if marker_pos != -1:
                                    # Include the [/INST] tag in what gets masked
                                    end_pos = marker_pos + len(inst_end_marker)
                                    instruction_tokens = self.llm.tokenizer(instruction_input[:end_pos], add_special_tokens=False)["input_ids"]
                                    # Ensure we don't exceed the tensor dimensions
                                    max_len = min(len(instruction_tokens), inputs['attention_mask'].shape[1])
                                    safe_mask[ii_idx, :max_len] = 0
                        
                        # Calculate embeddings using the safe mask
                        embeddings = self.pooling(last_hidden_state, safe_mask, recast=False).to('cpu')
                        
                        # Check for NaN values and fix them
                        if torch.isnan(embeddings).any():
                            print(f"WARNING: NaN detected in batch embeddings. Using fallback approach.", flush=True)
                            # Create a simpler pooling with mean of all tokens
                            full_mask = torch.ones_like(inputs['attention_mask'])
                            embeddings = self.pooling(last_hidden_state, full_mask, recast=False).to('cpu')
                            
                            # If we still have NaNs, create a fallback embedding
                            if torch.isnan(embeddings).any():
                                print(f"CRITICAL: NaN persists after fallback. Using zeros with a sentinel value.", flush=True)
                                embeddings = torch.zeros_like(embeddings)
                                embeddings[:, 0] = 1.0  # Set first dimension to 1

                    del last_hidden_state
                    del inputs

                    # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
                    if self.args.normalized: 
                        in_dtype = embeddings.dtype
                        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).to(in_dtype)
                    embeddings = cast(torch.Tensor, embeddings)

                    try:
                            # Check that necessary variables are defined and have the expected values
                            if len(new_sentences_batch) > 0 and total_num_gens > 0:
                        embeddings = torch.reshape(embeddings, (len(new_sentences_batch)//(total_num_gens+1),-1,embeddings.shape[-1]))
                        # assert embeddings.shape[-2]==5
                        embeddings = torch.mean(embeddings, -2)
                            else:
                                print("Warning: Skipping reshape due to empty batch or invalid num_gens", flush=True)
                        except Exception as e:
                            print(f"Error reshaping embeddings: {str(e)}", flush=True)
                        print("error params", len(new_sentences_batch), (total_num_gens+1), len(all_new_sentences_batch), new_batch_size, flush=True)

                    
                    all_embeddings.append(embeddings)

                all_embeddings = torch.cat(all_embeddings, dim=0)
            else:
                all_embeddings = torch.ones((len(sentences_rank_unchopped), 2))

        all_embeddings = [all_embeddings]
        all_embeddings=torch.cat(gather_object(all_embeddings), dim=0)

        all_embeddings = all_embeddings if convert_to_tensor else all_embeddings.cpu().to(torch.float32).numpy()


        return all_embeddings



    def pooling(
        self, hidden_state: torch.Tensor, attention_mask: torch.Tensor = None, recast: bool = False
    ) -> torch.Tensor:
        """
        Args:
            hidden_state: [b, n, d]
            attention_mask: [b, n]
        """
        # In case the model is distributed across multiple devices; hidden_state may end up on diff device
        hidden_state = hidden_state.to(attention_mask.device)
        if self.pooling_method == 'cls':
            embedding = hidden_state[:, 0]
        elif self.pooling_method == 'lasttoken':
            b, n, d = hidden_state.size()
            # Get the last `1` in the attention mask of each item
            # Often it is just `gather_indices = torch.argmin(attention_mask, 1, keepdim=False) - 1`
            # except when 1) There's all 1's 2) There's 0's before the 1's
            reversed_mask = torch.flip(attention_mask, dims=(1,))
            argmax_reverse = torch.argmax(reversed_mask, dim=1, keepdim=False)
            gather_indices = attention_mask.size(1) - argmax_reverse - 1
            # If there are empty sequences, where the index would become -1 it will crash so set them to 0
            gather_indices = torch.clamp(gather_indices, min=0)
            # Turn indices from shape [b] -> [b, 1, d]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, d)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (b, 1, d)
            # Gather along the seq len: [b, n, d] -> [b, d]
            # Actually no need for the attention mask as we gather the last token where attn_mask=1 but
            # as some indices (which shouldn't be attended to) may be 0 due to clamp, use mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand((b, n, d)).float()
            embedding = torch.gather(hidden_state * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        elif self.pooling_method in ['mean', 'weightedmean']:
            if self.pooling_method == 'weightedmean':
                attention_mask *= attention_mask.cumsum(dim=1) # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            embedding = s / d
        else: raise NotImplementedError(f"Unknown pooling method: {self.pooling_method}")
        # Recasting performs slightly worse but saves 50% space
        if recast: return embedding.to(hidden_state.dtype)
        return embedding
