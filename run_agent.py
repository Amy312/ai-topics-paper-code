import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool, RagTool

# Carga las variables de entorno
load_dotenv()

# Configuración del LLM
llm = LLM(model="gpt-4o-mini")
current_dir = os.path.dirname(os.path.abspath(__file__))

# Definimos el agente "Analyzer"
analyzer_agent = Agent(
    role="Analyzer",
    goal="Analyze documents and extract relevant information for coding implementation.",
    backstory="""You are an expert in analyzing technical papers and documentation. 
    You use various tools to gather, summarize, and prepare information for coding purposes.""",
    llm=llm,
)

# Herramientas (tools) para el agente "Analyzer"
paper_rag_tool = RagTool(source_type="pdf", file_path=os.path.join(current_dir, "paper.pdf"))
fastapi_docs_tool = RagTool(source_type="web", url="https://fastapi.tiangolo.com/")
scipy_docs_tool = RagTool(source_type="web", url="https://docs.scipy.org/doc/")

# Tareas (tasks) del agente "Analyzer"
analyze_paper_task = Task(
    description="Read and summarize key information from the technical paper for coding implementation.",
    expected_output="A concise summary with the technical details necessary for implementation.",
    agent=analyzer_agent,
    tools=[paper_rag_tool],
    output_file="paper_summary.txt"
)

fastapi_docs_task = Task(
    description="Extract relevant information from the FastAPI documentation.",
    expected_output="A detailed summary of FastAPI usage relevant to the implementation.",
    agent=analyzer_agent,
    tools=[fastapi_docs_tool],
    output_file="fastapi_summary.txt"
)

scipy_docs_task = Task(
    description="Extract relevant information from the SciPy documentation.",
    expected_output="A detailed summary of SciPy usage relevant to the implementation.",
    agent=analyzer_agent,
    tools=[scipy_docs_tool],
    output_file="scipy_summary.txt"
)

# Definimos el agente "Coder"
coder_agent = Agent(
    role="Coder",
    goal="Generate Python code using SciPy and FastAPI based on the provided technical paper information.",
    backstory="""You are a senior Python developer specialized in implementing code from technical specifications 
    using FastAPI and SciPy.""",
    allow_code_execution=True,
    llm=llm,
)

# Tarea del agente "Coder"
generate_code_task = Task(
    description="Generate Python code using the information from the paper, FastAPI, and SciPy documentation.",
    expected_output="A Python file implementing the code as described in the paper using FastAPI and SciPy.",
    agent=coder_agent,
    tools=[],  # No tools needed for this task as the agent is coding based on summaries
    output_file="implementation.py"
)

# Crew que contiene todos los agentes y tareas
dev_crew = Crew(
    agents=[analyzer_agent, coder_agent],
    tasks=[analyze_paper_task, fastapi_docs_task, scipy_docs_task, generate_code_task],
    verbose=True
)

# Ejecutar el crew
result = dev_crew.kickoff()

print(result)
