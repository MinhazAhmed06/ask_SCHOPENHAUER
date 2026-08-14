from dotenv import load_dotenv
from importlib.metadata import version
from langchain_openrouter import ChatOpenRouter

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")
print(core_version)
print(lg_version)


def main():
    llm = ChatOpenRouter(model="inclusionai/ling-3.0-flash:free")
    response = llm.invoke("say 'setup complete' in one word.")
    print(response)


if __name__ == "__main__":
    main()