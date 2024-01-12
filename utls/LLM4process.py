import transformers
import torch

class BasicLLM():
    def __init__(self) -> None:
        pass
    
    def _create_LLM(self):   
        raise NotImplementedError

    def _LLM_func(self, input_encodings):
        raise NotImplementedError
    
    def get_interaction_emb(self, user_list):
        raise NotImplementedError

    def general_LLM(self, input_text):
        raise NotImplementedError

class Llama2_70(BasicLLM):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.device = config["device"]
        self._create_LLM()

    def _create_LLM(self):
        token = '-------'
        model = "meta-llama/Llama-2-70b-chat-hf"
        self.pipeline = transformers.pipeline(
            task="feature-extraction",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token
        )
    
    def general_LLM(self, input_text):
        with torch.no_grad():
            result = self.pipeline(input_text, return_tensors=True)
        return result.detach().clone().squeeze().to(torch.float32)

    def _LLM_func(self, input_encodings):
        pass
    
    def get_interaction_emb(self, user):
        pass


class Llama2_13(BasicLLM):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.device = config["device"]
        self._create_LLM()

    def _create_LLM(self):
        token = '-----------'
        model = "meta-llama/Llama-2-13b-chat-hf"
        self.pipeline = transformers.pipeline(
            task="feature-extraction",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token
        )
    
    def general_LLM(self, input_text):
        with torch.no_grad():
            result = self.pipeline(input_text, return_tensors=True)
        return result.detach().clone().squeeze().to(torch.float32)

    def _LLM_func(self, input_encodings):
        pass
    
    def get_interaction_emb(self, user):
        pass

class Llama2(BasicLLM):
    def __init__(self, config) -> None:
        super().__init__()

        self.config = config
        self.device = config["device"]
        self._create_LLM()

    def _create_LLM(self):
        token = '-----------'
        model = "meta-llama/Llama-2-7b-chat-hf"
        self.pipeline = transformers.pipeline(
            task="feature-extraction",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
            token=token
        )
    
    def general_LLM(self, input_text):
        with torch.no_grad():
            result = self.pipeline(input_text, return_tensors=True)
        return result.detach().clone().squeeze().to(torch.float32)

    def _LLM_func(self, input_encodings):
        pass
    
    def get_interaction_emb(self, user):
        pass