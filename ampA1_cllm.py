#!/usr/bin/env python
# coding: utf-8

#
# see https://python.langchain.com/docs/modules/model_io/models/llms/custom_llm
#
from typing import Any, List, Mapping, Optional
from time import time

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

import requests, ast


class OCI_AmpereA1_LLM(LLM):
    debug: bool = True

    service_endpoint: str = "http://144.24.98.46:5005/api/chat/"
    model: str = "Llama-Pro-8B-Instruct" 
    headers: dict = {'accept': 'application/json'}
    temperature: float = 0.1
    top_k: int = 50
    top_p: float = 0.95
    max_length: int = 2048
    context_window: int = 2048
    gpu_layers: int = 0
    repeat_last_n: int = 64
    repeat_penalty: float = 1.3
    init_prompt: str = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    n_threads: int = 4

    chat_id: Optional[str]
    new_chat_endpoint: Optional[str]

    timeout: Optional[int] = 10

    """OCI Ampere A1 LLM model.

    Example:
        .. code-block:: python

            a1_llm = OCI_AmpereA1_LLM() # any of the Class params can be passed to overwrite default values
            a1_llm._llm_type
            for i in a1_llm:
                print(i) 
            a1_llm.invoke('who is the current President of USA?')
            a1_llm.get_all_chats()
            a1_llm.delete_all_chats()
            a1_llm.chat_history()

    """

    def new_chat(self):
        print('### Calling new_chat() ###')
         # calling OCI Ampere A1 LLM
        tStart = time()

        params = {
            "model": self.model, 
            "temperature": self.temperature, 
            "top_k": self.top_k,
            "top_p": self.top_p, 
            "max_length": self.max_length, 
            "context_window": self.context_window, 
            "gpu_layers": self.gpu_layers, 
            "repeat_last_n": self.repeat_last_n, 
            "repeat_penalty": self.repeat_penalty, 
            "init_prompt": self.init_prompt, 
            "n_threads": self.n_threads
        }

            
        print("Calling OCI Ampere A1 LLM")
        response = requests.post(self.service_endpoint, params=params, headers=self.headers)

        tEla = time() - tStart

        if self.debug:
            print(f"Elapsed time: {round(tEla, 1)} sec...")
            print()

        print(f'Response from within new_chat(): {response}')
        self.chat_id = response.json() # new chat id
        print(f'Chat ID: {self.chat_id}')
        self.new_chat_endpoint = f'{self.service_endpoint}{self.chat_id}/question'
        print(f'Chat Endpoint: {self.new_chat_endpoint}')
        print('### Ending new_chat() ###')

    def __init__(self, **kwargs):
        print('### Calling __init__() ###')
        super().__init__(**kwargs)
        model = self.model
        service_endpoint=self.service_endpoint
        headers = self.headers
        # self.new_chat()
        # self.new_chat_endpoint = f'{service_endpoint}{self.chat_id}/question'
        # print(f'Chat Endpoint: {self.new_chat_endpoint}')
        print('### Ending __init__() ###')

    @property
    def _llm_type(self) -> str:
        return "OCI Ampere A1 LLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        
        print('### Calling _call() ###')
        
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        if self.debug:
            print()
            print("The input prompt is:")
            print(prompt)
            print()
    
        self.new_chat()
        # self.new_chat_endpoint = f'{self.service_endpoint}{self.chat_id}/question'
        # print(f'Chat Endpoint: {self.new_chat_endpoint}')
    
        params = {
            'chat_id': self.chat_id, 
            'prompt': [prompt]
        }

        # print(self.new_chat_endpoint)

        response = requests.post(
            self.new_chat_endpoint,
            params=params,
            headers=self.headers,
        )

        print(f'Response from within _call(): {response}')
        json_resp = response.json()
        if self.debug:
            print(f'Original json response: {json_resp}')

        # had to use ast as response is not in valid json format which requires properties and string values to be in double quotes
        resp_dict = ast.literal_eval(json_resp)
        if self.debug:
            print(f'AST converted json response: {resp_dict}')

        # print(f"Full response: {response.json()}")
        
        print('### Ending _call() ###')
        return resp_dict['choices'][0]['text'].strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "chat_endpoint": self.new_chat_endpoint, 
            "model": self.model, 
            "temperature": self.temperature, 
            "top_k": self.top_k,
            "top_p": self.top_p, 
            "max_length": self.max_length, 
            "context_window": self.context_window, 
            "gpu_layers": self.gpu_layers, 
            "repeat_last_n": self.repeat_last_n, 
            "repeat_penalty": self.repeat_penalty, 
            "init_prompt": self.init_prompt, 
            "n_threads": self.n_threads
        }
    
    def __repr__(self):
        return f'Model Parameters:\n{self._identifying_params}'
    
    def get_all_chats(self):
        # Get all chats
        print(f'Below are the different Chat Ids part of current LLM instance')
        response = requests.get(self.service_endpoint, headers=self.headers)
        print(response)
        for i, item in enumerate(response.json()):
            print(f'{i+1} :: {item}')

    def delete_all_chats(self):
        # delete all chats
        print(f'Deleting all Chat Ids part of current LLM instance')
        delete_chats_endpoint = f'{self.service_endpoint}delete/all'
        response = requests.delete(delete_chats_endpoint, headers=self.headers)
        print(response)
        print(response.json())

    def chat_history(self):
        print(f'Below is the history of chats within Chat ID: {self.chat_id}')
        chat_history_endpoint = f'{self.service_endpoint}{self.chat_id}/history'
        response = requests.get(chat_history_endpoint, headers=self.headers)
        print(response)
        # print(response.json())
        for i, item in enumerate(response.json()):
            print(f'{i+1} :: {item}')