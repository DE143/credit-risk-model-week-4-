# Credit Risk Probability Model for Alternative Data

## Credit Scoring Business Understanding

### Basel II Accord and Model Interpretability
The Basel II Capital Accord emphasizes rigorous risk measurement and management. This influences our need for:
- **Transparent Models**: Regulators require understanding of how decisions are made
- **Documentation**: Comprehensive documentation of model development, validation, and performance
- **Risk Weighted Assets**: Models must accurately calculate risk weights for capital requirements
- **Governance**: Clear model governance and regular validation are mandated

These requirements make interpretable models crucial for regulatory compliance and stakeholder trust.

### Proxy Variable Necessity and Risks
Since we lack direct default labels, creating a proxy is essential but introduces risks:

**Why Necessary:**
- No direct credit history in e-commerce data
- Need to infer creditworthiness from behavioral patterns
- Enables model training with available data

**Potential Business Risks:**
- **Proxy Mismatch**: Behavioral patterns may not perfectly correlate with credit default
- **Selection Bias**: Model may learn spurious correlations in the proxy
- **Regulatory Scrutiny**: Regulators may question proxy validity
- **Performance Drift**: Proxy-target relationship may change over time

### Model Complexity Trade-offs

**Simple Models (Logistic Regression with WoE):**
- **Advantages**: 
  - High interpretability (coefficients = feature importance)
  - Regulatory compliance friendly
  - Easier to validate and explain
  - Lower risk of overfitting
- **Disadvantages**:
  - May capture fewer complex patterns
  - Lower predictive power on non-linear relationships

**Complex Models (Gradient Boosting):**
- **Advantages**:
  - Higher predictive accuracy
  - Captures complex, non-linear patterns
  - Handles feature interactions automatically
- **Disadvantages**:
  - Lower interpretability ("black box" problem)
  - Higher regulatory scrutiny
  - More difficult to validate
  - Higher risk of overfitting

**Recommended Approach**: Use interpretable models (Logistic Regression) for initial deployment with complex models for validation. Consider model ensembles or SHAP values for interpretability with complex models.