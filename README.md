# Documentation claim auditor

This offline tool checks whether documentation claims about salient-term harvesting agree with static patterns in Python source. It uses only the standard library, makes no network requests, and does not modify the scanned repository.

## Usage

Install the distribution, then run:

```sh
keyword-extraction-claim-audit audit .
python3 -m keyword_extraction_claim_audit audit . --json
keyword-extraction-claim-audit audit . --ignore tests/ --fail-on medium
```

The root README and Markdown files under docs are checked; fenced blocks are skipped and inline code is included. Python files are inspected for evidence. Repeat the ignore option for additional path prefixes. Verbose mode reports scope and decisions on standard error. The default threshold is high; medium findings cause failure when selected. Exit status is zero for a pass, two for a finding at the threshold, and one for usage or I/O errors.

Finding identifiers:

```
documented-keyword-extraction-claim-false
documented-no-keyword-extraction-claim-false
```

## Differentiation

These neighboring tools cover distinct tasks:

```
tokenization-claim-audit
stemming-claim-audit
lemmatization-claim-audit
named-entity-recognition-claim-audit
summarization-claim-audit
```

In order, they check text splitting, suffix stripping, dictionary base forms, typed entity spans, and passage condensing. This tool checks salient-term harvesting.

## Limitations

This is static pattern matching, not semantic judgment. It recognizes exact substrings rather than paraphrases. The following unrelated programming phrase collides with a bare matching substring and can cause a high finding:

```
keyword arguments
keyword argument
```

A generic call on an unrelated object can count as evidence because its method name is shared:

```python
object.extract_keywords(text)
```

Vectorizer feature-name read-back counts even when the terms are used only for display. Patterns are checked within individual Python modules; dynamic behavior and other languages are not understood. Evidence in tests can also count as source evidence.
