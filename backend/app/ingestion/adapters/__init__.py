"""Provider adapters. Each turns a file into CanonicalTransactions and does
nothing else: no normalisation beyond reading cells, no validation, no
storage. Adding a provider means adding an adapter, not a pipeline."""
