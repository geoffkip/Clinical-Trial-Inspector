import json
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from modules.tools import get_study_details
from modules.utils import load_environment

# Load env for API key
load_environment()

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

EXTRACT_PROMPT = PromptTemplate(
    template="""
    You are a Clinical Informatics Expert.
    Your task is to extract structured cohort requirements from the following Clinical Trial Eligibility Criteria.

    Output a JSON object with two keys: "inclusion" and "exclusion".
    Each key should contain a list of rules.
    Each rule should have:
    - "concept": The medical concept (e.g., "Type 2 Diabetes", "Metformin").
    - "domain": The domain (Condition, Drug, Measurement, Procedure, Observation).
    - "temporal": Any temporal logic (e.g., "History of", "Within last 6 months").
    - "codes": A list of potential ICD-10 or RxNorm codes (make a best guess).

    CRITERIA:
    {criteria}

    JSON OUTPUT:
    """,
    input_variables=["criteria"],
)

SQL_PROMPT = PromptTemplate(
    template="""
    You are a SQL Expert specializing in Healthcare Claims Data Analysis.
    Generate a standard SQL query to define a cohort of patients based on the following structured requirements.

    ### Schema Assumptions
    1.  **`medical_claims`** (Diagnoses & Procedures):
        - `patient_id`, `claim_date`, `diagnosis_code` (ICD-10), `procedure_code` (CPT/HCPCS).
    2.  **`pharmacy_claims`** (Drugs):
        - `patient_id`, `fill_date`, `ndc_code`.

    ### Logic Rules
    1.  **Conditions (Diagnoses)**:
        - Require **at least 2 distinct claim dates** where the diagnosis code matches.
        - These 2 claims must be **at least 30 days apart** (to confirm chronic condition).
    2.  **Drugs**:
        - Require at least 1 claim with a matching NDC code.
    3.  **Procedures**:
        - Require at least 1 claim with a matching CPT/HCPCS code.
    4.  **Exclusions**:
        - Exclude patients who have ANY matching claims for exclusion criteria.

    ### Requirements (JSON)
    {requirements}

    ### Output
    Generate a single SQL query that selects `patient_id` from the claims tables meeting the criteria.
    Use Common Table Expressions (CTEs) for clarity.
    Do NOT output markdown formatting (```sql), just the raw SQL.

    SQL QUERY:
    """,
    input_variables=["requirements"],
)


def extract_cohort_requirements(criteria_text: str) -> dict:
    """Uses LLM to parse criteria text into structured JSON."""
    chain = EXTRACT_PROMPT | llm
    response = chain.invoke({"criteria": criteria_text})
    try:
        # Clean up potential markdown code blocks
        text = response.content.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM output", "raw_output": response.content}


def generate_cohort_sql(requirements: dict) -> str:
    """Uses LLM to translate structured requirements into SQL."""
    chain = SQL_PROMPT | llm
    response = chain.invoke({"requirements": json.dumps(requirements, indent=2)})
    return response.content.replace("```sql", "").replace("```", "").strip()


@tool("get_cohort_sql")
def get_cohort_sql(nct_id: str) -> str:
    """
    Generates a SQL query to define the patient cohort for a specific study (NCT ID).
    
    Args:
        nct_id (str): The ClinicalTrials.gov identifier (e.g., NCT01234567).
        
    Returns:
        str: A formatted string containing the Extracted Requirements (JSON) and the Generated SQL.
    """
    # 1. Fetch Study Details
    # We reuse the existing tool logic to get the text
    study_text = get_study_details(nct_id)
    
    if "No study found" in study_text:
        return f"Could not find study {nct_id}."

    # 2. Extract Requirements
    requirements = extract_cohort_requirements(study_text)
    
    # 3. Generate SQL
    sql_query = generate_cohort_sql(requirements)
    
    return f"""
### 📋 Extracted Cohort Requirements
```json
{json.dumps(requirements, indent=2)}
```

### 💾 Generated SQL Query (OMOP CDM)
```sql
{sql_query}
```
"""
