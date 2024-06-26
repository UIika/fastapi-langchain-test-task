from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain_huggingface import HuggingFacePipeline

# language model creation
llm = HuggingFacePipeline.from_model_id(
    model_id="facebook/bart-large-cnn",
    task="summarization",
)

# pydantic schema for validating user text input
class TextSchema(BaseModel):
    text: str


app = FastAPI(title='FastAPI Langchain test task')

# endpoint for summarizing
@app.post("/summarize", tags=['summarize'])
async def summarize(text_schema: TextSchema):
    summary = llm.invoke(text_schema.text)
    return {"summary": summary}

# hosting uvicorn server if this module is executed
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)