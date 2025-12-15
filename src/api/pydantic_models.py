from pydantic import BaseModel

class CreditRequest(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    Amount: float
    Value: float
    txn_hour: int
    txn_day: int
    txn_month: int
    txn_year: int
    CurrencyCode: str
    CountryCode: str
    ProviderId: str
    ProductId: str
    ProductCategory: str
    ChannelId: str
    PricingStrategy: str

class CreditResponse(BaseModel):
    risk_probability: float
