"""Statement ingestion: provider adapters -> one canonical transaction model.

Adapters only parse and map. Everything after canonicalisation - identity
normalisation, validation, deduplication, storage, payee history, graph and
profile - is shared, so a CSV and a PDF of the same payments reach the risk
engine as the same facts.
"""
