from pydantic import BaseModel, Field


class CritiqueModel(BaseModel):
    superfluous: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class ResponderOutputModel(BaseModel):
    response: str
    critique: CritiqueModel
    search: list[str] = Field(default_factory=list)


class RevisedOutputModel(BaseModel):
    response: str
    critique: CritiqueModel
    search: list[str] = Field(default_factory=list)
