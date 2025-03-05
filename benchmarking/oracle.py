from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from openai import OpenAI

# openai apis
MODEL_GPT4o_MINI = 'gpt-4o-mini'
MODEL_GPT4o = 'gpt-4o'
MODEL_GPT4_TURBO = 'gpt-4-turbo'
# MODEL_GPT4 = 'gpt-4'                  # out-dated, should not use
# MODEL_GPT3_5 = 'gpt-3.5-turbo-0125'   # out-dated, should not use
MODEL_EMBED_SMALL = 'text-embedding-3-small'
# google models
MODEL_GEMINI_15_PRO = 'gemini-1.5-pro'
MODEL_GEMINI_15_FLASH = 'gemini-1.5-flash'
MODEL_GEMINI_1_PRO = 'gemini-1.0-pro'
MODEL_EMBED_GOOGLE = 'text-embedding-004'
# azure deployments (for phi-family models you need to specify the endpoint url by yourself)
MODEL_PHI_3_MINI = 'phi-3-mini'
MODEL_PHI_3_5_MINI = 'phi-3.5-mini'
MODEL_PHI_3_SMALL = 'phi-3-small'
MODEL_PHI_3_MEDIUM = 'phi-3-medium'
# deepinfra deployments
DEEP_INFRA_BASE_URL = 'https://api.deepinfra.com/v1/openai'
MODEL_LLAMA_3_8B = 'llama-3-8B'
MODEL_LLAMA_3_70B = 'llama-3-70B'
MODEL_MIXTRAL_8X7B = 'mixtral-8x7B'
DEEP_INFRA_MAP = {MODEL_LLAMA_3_8B: 'meta-llama/Meta-Llama-3-8B-Instruct',
                  MODEL_LLAMA_3_70B: 'meta-llama/Meta-Llama-3-70B-Instruct',
                  MODEL_MIXTRAL_8X7B: 'mistralai/Mixtral-8x7B-Instruct-v0.1',}

openai_model_list = [MODEL_GPT4o_MINI, MODEL_GPT4o, MODEL_GPT4_TURBO, MODEL_EMBED_SMALL]
google_model_list = [MODEL_GEMINI_15_PRO, MODEL_GEMINI_15_FLASH, MODEL_GEMINI_1_PRO, MODEL_EMBED_GOOGLE]
azure_model_list = [MODEL_PHI_3_MINI, MODEL_PHI_3_5_MINI, MODEL_PHI_3_SMALL, MODEL_PHI_3_MEDIUM]
deepinfra_model_list = [MODEL_LLAMA_3_8B, MODEL_LLAMA_3_70B, MODEL_MIXTRAL_8X7B]
all_model_list = []
for l in [openai_model_list, google_model_list, azure_model_list, deepinfra_model_list]:
    all_model_list += l

def response_failure(prompt_user, model, e,):
    print(f"err: The following error occurred when querying {prompt_user} through {model}:")
    print(e)
    return {"query": prompt_user, "answer": "QUERY_FAILED"}

def openai_chat_query(
        # single query
        client, # openai client object;
        model,
        prompt_sys,
        prompt_user,
        temp,
    ) -> dict:

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    # {"role": "system", "content": prompt_sys},
                    {"role": "user", "content": prompt_user},
                ],
                stream=False,
                temperature=temp,
            )

            response_result = ""
            # for chunk in stream:
            if completion.choices[0].message:
                response_result += completion.choices[0].message.content

            return {"query": prompt_user, "answer": response_result}

        except Exception as e:  # Consider capturing a specific exception if possible
            return response_failure(prompt_user, model, e)

def openai_embed_query(client, model, list_of_text, dimensions) -> dict:
    # small batch query
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = client.embeddings.create(input=list_of_text, model=model).data # , dimensions=dimensions
    
    res = {list_of_text[i]:data[i].embedding for i in range(len(list_of_text))}
    return res

def google_chat_query(client, model, prompt_sys, prompt_user, temp,) -> dict:
    # here model and prompt_sys are useless, just to align with the openai interface
    gen_config = {"temperature": temp,}
    try:
        response = client.generate_content(prompt_user, generation_config=gen_config)
        res =  {"query": prompt_user, "answer": response.text}
        return res
    except Exception as e:
        return response_failure(prompt_user, model, e)
    
# def azure_chat_query(client, model, prompt_sys, prompt_user, temp,) -> dict:
#     sys_message = SystemMessage(content=prompt_sys)
#     usr_message = UserMessage(content=prompt_user)
#     try:
#         response = client.complete(messages=[sys_message, usr_message,], temperature=temp,)
#         response_text = response['choices'][0]['message']['content']
#         res = {"query": prompt_user, "answer": response_text}
#         return res
#     except Exception as e:
#         return response_failure(prompt_user, model, e)

class Oracle:
    def __init__(self, model, apikey=None, azure_end_point=''):
        assert model in all_model_list, f'err: model named {model} is not supported'
        self.model = model
        self.apikey = apikey
        # for openai models
        if model in openai_model_list:
            if apikey: openai.api_key = apikey
            self.client = OpenAI(api_key=apikey)
        elif model in deepinfra_model_list:
            self.client = OpenAI(api_key=apikey, base_url=DEEP_INFRA_BASE_URL)
        # elif model in google_model_list:
        #     genai.configure(api_key=apikey,)
        #     self.client = genai.GenerativeModel(model)
        # elif model in azure_model_list:
        #     azure_credential = AzureKeyCredential(apikey)
        #     self.client = ChatCompletionsClient(endpoint=azure_end_point, credential=azure_credential)
    
    def query(self, prompt_sys, prompt_user, temp=0.1,):
        if self.model in openai_model_list:
            return openai_chat_query(self.client, self.model, prompt_sys, prompt_user, temp)
        elif self.model in deepinfra_model_list:
            return openai_chat_query(self.client, DEEP_INFRA_MAP[self.model], prompt_sys, prompt_user, temp)
        # elif self.model in google_model_list:
        #     # prompt_sys not supported
        #     return google_chat_query(self.client, self.model, prompt_sys, prompt_user, temp)
        # elif self.model in azure_model_list:
        #     return azure_chat_query(self.client, self.model, prompt_sys, prompt_user, temp)
    
    def query_all(self, prompt_sys, prompt_user_all, workers=12, temp=0.1, **kwargs):
        # prompt_user_all: [str,]

        # collect procedure
        results = []
        print(f"Total queries: {len(prompt_user_all)}, start collecting...")

        model_name = self.model
        if self.model in openai_model_list:
            single_query_fn = openai_chat_query
        elif self.model in deepinfra_model_list:
            single_query_fn = openai_chat_query
        #     model_name = DEEP_INFRA_MAP[self.model]
        # elif self.model in google_model_list:
        #     single_query_fn = google_chat_query
        #     if prompt_sys: # refresh the system prompt
        #         self.client = genai.GenerativeModel(model_name=self.model, system_instruction=prompt_sys,)
        # elif self.model in azure_model_list:
        #     single_query_fn = azure_chat_query
        
        with ThreadPoolExecutor(max_workers=workers) as executor: # avg. 0.13s per item
            future_results = [executor.submit(single_query_fn, self.client, model_name, prompt_sys, p, temp) for p in prompt_user_all]
        
            for future in tqdm(as_completed(future_results), total=len(prompt_user_all), desc="Processing Items"):
                result = future.result()
                results.append(result)
        return results
    
    def encode(self, text, dim=1536):
        # single
        if self.model in openai_model_list:
            return openai_embed_query(self.client, self.model, [text], dim)

    def encode_all(self, list_of_text, workers, dim=1536, chunk_size=100):
        # split the list into small sublists
        sublists = [list_of_text[i:i + chunk_size] for i in range(0, len(list_of_text), chunk_size)]
        
        results = []
        print(f"Total queries: {len(list_of_text)}, start collecting...")
        
        if self.model in openai_model_list:
            single_query_fn = openai_embed_query

        with ThreadPoolExecutor(max_workers=workers) as executor: # avg. 0.13s per item
            future_results = [executor.submit(single_query_fn, self.client, self.model, p, dim) for p in sublists]
        
            for future in tqdm(as_completed(future_results), total=len(sublists), desc="Processing Items"):
                result = future.result()
                results.append(result)
        return results
    
    def get_model_info(self):
        model_info = {}
        if self.model in openai_model_list:
            model_info['model_details'] = self.client.models.retrieve(self.model)
        else:
            model_info['model_details'] = "No model info available for this model."
        
        model_info['api_key'] = self.apikey if self.apikey else "No API key provided."
        
        return model_info