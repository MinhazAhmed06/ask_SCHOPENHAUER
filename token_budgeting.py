import tiktoken

class TokenBudget:

    def __init__(self, max_token_per_request: int=5000):
        self.max_per_request = max_token_per_request
        self.usage = {
            "total_input":0,
            "total_output":0,
            "requests":0
        }

    def token_count(self, text:str) -> int:
        enc = tiktoken.encoding_for_model(model_name='gpt-4')
        token = len(enc.encode(text))
        return token

    def check_budget(self, text:str) -> tuple[bool, int]:
        tokens = self.token_count(text)
        return tokens <= self.max_per_request, tokens

    def record_usage(self, input_tokens: int, output_tokens: int):
        """Record token usage."""
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    