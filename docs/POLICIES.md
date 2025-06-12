# Policy Configuration Guide (`policies.yaml`)

This document provides a comprehensive guide to configuring the `policies.yaml` file for the Trylon AI Gateway. This file is the central control panel for defining all security, content, and operational guardrails.

A well-configured policy file allows you to precisely control the behavior of your AI applications, ensuring they are safe, compliant, and aligned with your business rules.

## Core Concepts

A **policy** is a single rule that the gateway checks for every message it processes. Each policy has:
- A specific **type** of check to perform (e.g., PII detection, toxicity analysis).
- A **condition** for triggering a violation (e.g., a confidence score above a threshold).
- An **action** to take when a violation occurs (e.g., block, observe, redact).
- **Scope** to determine if it applies to user input, LLM output, or both.

All policies are defined as a list under the `policies` key in your `policies.yaml` file.

## Policy Object Structure

Each object in the `policies` list has the following structure. Fields marked with `*` are required.

- `id`* (`integer`): **The type of policy.** This is a numeric ID that maps to a specific validator. See the [PolicyType ID Mapping](#policytype-id-mapping) table below.
- `name`* (`string`): A human-readable name for the policy, used in logs.
- `state`* (`boolean`): Enables (`true`) or disables (`false`) the policy. Disabled policies are completely ignored.
- `is_user_policy` (`boolean`, default: `true`): If `true`, the policy is applied to user and system prompts (input sent *to* the LLM).
- `is_llm_policy` (`boolean`, default: `true`): If `true`, the policy is applied to assistant responses (output received *from* the LLM).
- `action` (`integer`, default: `0`): The action to take upon violation. See the [Action Mapping](#action-mapping) table.
- `message` (`string`): The message to return in the response body and logs when the policy is violated. If not provided, a default message is used.
- `threshold` (`float`, 0.0-1.0): A general confidence score threshold for model-based policies (Toxicity, NER). Specific thresholds below will override this.
- `metadata` (`dict`, default: `{}`): An optional field for your own annotations, like `owner: "security-team"` or `ticket: "SEC-123"`.

### Policy-Specific Fields

- **For PII Leakage (`id: 1`)**
  - `pii_categories` (`list[string]`): A list of high-level Presidio categories to scan for (e.g., `DEFAULT`, `FINANCIAL`, `US_SPECIFIC`).
  - `pii_entities` (`list[string]`): A list of specific Presidio entity types (e.g., `EMAIL_ADDRESS`, `PHONE_NUMBER`, `CREDIT_CARD`).
  - `pii_threshold` (`float`, default: `0.5`): The minimum confidence score for an entity to be considered a violation.
- **For Prompt Leakage (`id: 2`)**
  - `protected_prompts` (`list[string]`): A list of sensitive strings, keywords, or secrets to detect.
  - `prompt_leakage_threshold` (`float`, default: `0.85`): The similarity score (0.0 to 1.0) for the fuzzy match. Higher values require a closer match.
- **For NER-based Checks (`id: 3, 4, 5`)**
  - `competitors` (`list[string]`): For Competitor Check (`id: 3`).
  - `persons` (`list[string]`): For Person Check (`id: 4`).
  - `locations` (`list[string]`): For Location Check (`id: 5`).
- **For Toxicity (`id: 6`)**
  - The `threshold` field is used to determine the cutoff for the profanity classification model.

## Detailed Mappings

### PolicyType `id` Mapping
This table shows which `id` corresponds to which validator.

| ID | PolicyType         | Description                                                                               | Key Fields                                                                |
|----|--------------------|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| 1  | `PII_LEAKAGE`      | Detects Personally Identifiable Information using Presidio.                               | `pii_categories`, `pii_entities`, `pii_threshold`                         |
| 2  | `PROMPT_LEAKAGE`   | Detects secrets or keywords in responses using fuzzy matching.                            | `protected_prompts`, `prompt_leakage_threshold`                           |
| 3  | `COMPETITOR_CHECK` | Detects mentions of specific organizations using a local NER model.                       | `competitors`, `threshold`                                                |
| 4  | `PERSON_CHECK`     | Detects mentions of specific people using a local NER model.                              | `persons`, `threshold`                                                    |
| 5  | `LOCATION_CHECK`   | Detects mentions of specific locations using a local NER model.                           | `locations`, `threshold`                                                  |
| 6  | `PROFANITY`        | Detects toxic or profane language using a local classification model.                     | `threshold`                                                               |

### Action Mapping
This table defines the `action` to be taken when a policy is violated.

| Value | Action     | Description                                                                                                                   |
|-------|------------|-------------------------------------------------------------------------------------------------------------------------------|
| 0     | `OVERRIDE` | **Block the interaction.** The proxy returns a 200 OK with a modified body indicating a block. This is the default and safest action. |
| 1     | `OBSERVE`  | **Log the violation but allow the interaction.** The original request/response proceeds. Ideal for monitoring or tuning policies.  |
| 2     | `REDACT`   | **Modify content.** (Note: Currently only supported by the `/safeguard` PII policy. Falls back to `OVERRIDE` in proxy mode.)     |
| 3     | `RETRY`    | Suggests a retry. Typically used for internal errors. Not recommended for general policy actions.                               |

### SafetyCode Mapping
When a policy is violated, the gateway returns a `SafetyCode` in the response headers (`X-Trylon-Safety-Code`) and logs.

| Code  | Meaning                 | Corresponding Policy ID(s) |
|-------|-------------------------|----------------------------|
| 0     | `SAFE`                  | N/A                        |
| 10    | `PII_DETECTED`          | 1                          |
| 20    | `PROMPT_LEAKED`         | 2                          |
| 30    | `COMPETITOR_DETECTED`   | 3                          |
| 40    | `PERSON_DETECTED`       | 4                          |
| 50    | `LOCATION_DETECTED`     | 5                          |
| 60    | `PROOFANE`              | 6                          |
| -10   | `GENERIC_UNSAFE`        | (e.g., bad request format) |
| -70   | `UNEXPECTED`            | (Internal Server Error)    |
| -80   | `TIMEOUT`               | (Request timed out)        |

---

## Policy Examples

### Example 1: Blocking PII in User Prompts
This policy blocks any user input containing an email address or U.S. Social Security Number with high confidence. It does not check LLM responses.

```yaml
- id: 1
  name: "Block PII from Users"
  state: true
  is_user_policy: true  # Only check user input
  is_llm_policy: false # Do not check LLM output
  action: 0             # Block the request
  message: "Your request was blocked because it contained sensitive information (PII)."
  pii_entities:
    - EMAIL_ADDRESS
    - US_SSN
  pii_threshold: 0.85
```

### Example 2: Observing Toxicity in LLM Responses
This policy will log a warning if an LLM response is flagged as toxic, but it will not stop the response from reaching the user. This is great for monitoring model behavior without impacting user experience.

```yaml
- id: 6
  name: "Monitor LLM Toxicity"
  state: true
  is_user_policy: false # Do not check user input
  is_llm_policy: true   # Only check LLM output
  action: 1             # Observe only
  message: "Potentially toxic content detected in LLM response."
  threshold: 0.75
```

### Example 3: Preventing Leaks of Internal Project Names
This policy prevents the LLM from accidentally mentioning internal codenames in its responses.

```yaml
- id: 2
  name: "Prevent Internal Codename Leaks"
  state: true
  is_user_policy: false
  is_llm_policy: true
  action: 0
  message: "This response was blocked as it contained confidential information."
  protected_prompts:
    - "Project Titan"
    - "Bluebird Initiative"
    - "Internal-API-Key-v2"
  prompt_leakage_threshold: 0.90
```

## Full `policies.yaml` Example
Here is a complete example file demonstrating a mix of active and inactive policies.

```yaml
policies:
  # ----------------------------------------------------------------
  # PII & Data Leakage Policies
  # ----------------------------------------------------------------
  - id: 1
    name: "Block High-Confidence PII (Email, Phone, SSN)"
    state: true
    is_user_policy: true
    is_llm_policy: true
    action: 0 # OVERRIDE
    message: "Interaction blocked due to sensitive data (PII)."
    pii_entities:
      - EMAIL_ADDRESS
      - PHONE_NUMBER
      - US_SSN
    pii_threshold: 0.75

  - id: 2
    name: "Prevent Secret Key Leakage from LLM"
    state: true
    is_user_policy: false
    is_llm_policy: true
    action: 0 # OVERRIDE
    message: "Response blocked for security reasons."
    protected_prompts:
      - "sk-internal-..." # A pattern for an internal key
      - "Project-QuantumLeap"
    prompt_leakage_threshold: 0.95

  # ----------------------------------------------------------------
  # Content & Behavior Policies
  # ----------------------------------------------------------------
  - id: 6
    name: "Observe Profanity"
    state: true
    is_user_policy: true
    is_llm_policy: true
    action: 1 # OBSERVE
    message: "Profane language was detected."
    threshold: 0.8

  - id: 3
    name: "Block Competitor Mentions by LLM"
    state: true
    is_user_policy: false
    is_llm_policy: true
    action: 0 # OVERRIDE
    message: "This response was modified to remove competitor names."
    competitors:
      - "Acme Corporation"
      - "Global-Tech Inc."
    threshold: 0.8

  # ----------------------------------------------------------------
  # Inactive / Future Policies
  # ----------------------------------------------------------------
  - id: 4
    name: "Monitor for Executive Mentions (Inactive)"
    state: false # This policy is currently turned off
    is_user_policy: true
    is_llm_policy: true
    action: 1 # OBSERVE
    persons:
      - "Jane Doe"
      - "John Smith"
    threshold: 0.9
```

## Best Practices

1.  **Start in `OBSERVE` Mode:** When introducing a new policy, always start with `action: 1` (OBSERVE). Monitor your logs to see what it flags. This prevents false positives from disrupting your application. Once you are confident in the policy's accuracy and have tuned its threshold, switch it to `action: 0` (OVERRIDE).
2.  **Use Version Control:** Your `policies.yaml` is a critical part of your application's logic. Store it in Git (or your preferred VCS) to track changes, review updates, and easily roll back if needed.
3.  **Be Specific:** A few highly specific, high-threshold policies are often more effective than many broad, low-threshold policies that can lead to false positives. For example, blocking specific `pii_entities` is usually better than blocking the `DEFAULT` category.
4.  **Tailor `is_user_policy` and `is_llm_policy`:** Think about the direction of risk. You might want to be very strict about PII from users (`is_user_policy: true`) but more lenient on PII from an LLM that is *supposed* to return contact information (`is_llm_policy: false`).