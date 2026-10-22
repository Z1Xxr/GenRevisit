# Development

Install the test dependencies and run the suite from the repository root:

```bash
python -m pip install -e ".[test]"
python -m pytest tests -q
```

For agent interface checks only:

```bash
python -m pytest tests/test_agent_contract.py -q
```

[Back to GenRevisit](../README.md)
