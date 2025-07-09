from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, load_tool, tool
import datetime
import pytz
import yaml
import os
from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get('SECRET_KEY')



# ------------------------------- Custom tools ---------------------------------

@tool
def my_custom_tool(arg1: str, arg2: int) -> str:
    """Example utility that echoes its inputs.

    Args:
        arg1 (str): A string you want echoed back.
        arg2 (int): A number that will also be echoed back.

    Returns:
        str: A sentence containing the provided *arg1* and *arg2*.
    """
    return f"Echo: {arg1} (number {arg2})"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """Get the current local time for a given timezone.

    Args:
        timezone (str): IANA timezone identifier, e.g., "Africa/Lagos" or "America/New_York".

    Returns:
        str: Current time formatted as "YYYY-MM-DD HH:MM:SS" in the requested timezone.
            If the timezone is invalid, an error message is returned instead.
    """
    try:
        tz = pytz.timezone(timezone)
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return local_time
    except Exception as e:
        return f"Error fetching time for '{timezone}': {e}"


# ----------------------------- Model wrapper ----------------------------------

class SafeInferenceClientModel(InferenceClientModel):
    """InferenceClientModel that coerces token counters to integers.

    Some inference endpoints leave *last_input_token_count* or *last_output_token_count*
    as **None**.  This subclass patches them to **0** after every call so downstream
    code (e.g. Gradio_UI) can safely perform arithmetic.
    """

    def _ensure_token_counters(self):
        if getattr(self, "last_input_token_count", None) is None:
            self.last_input_token_count = 0
        if getattr(self, "last_output_token_count", None) is None:
            self.last_output_token_count = 0

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        self._ensure_token_counters()
        return result

    def chat(self, *args, **kwargs):
        result = super().chat(*args, **kwargs)
        self._ensure_token_counters()
        return result


# --------------------------------- Model --------------------------------------

model = SafeInferenceClientModel(
    max_tokens=2096,
    temperature=0.5,
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    custom_role_conversions=None,
)

# ----------------------------- Prompt templates --------------------------------

with open("prompts_v2.yaml", "r", encoding="utf-8") as f:
    prompt_templates = yaml.safe_load(f)

# --------------------------------- Agent --------------------------------------

final_answer_tool = FinalAnswerTool()

agent = CodeAgent(
    model=model,
    tools=[final_answer_tool, my_custom_tool, get_current_time_in_timezone],
    max_steps=6,
    verbosity_level=1,
    prompt_templates=prompt_templates,
)

# ---------------------------------- UI ----------------------------------------

if __name__ == "__main__":
    GradioUI(agent).launch()
