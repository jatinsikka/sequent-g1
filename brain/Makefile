PY=python

.PHONY: build-index plan exec test

build-index:
	$(PY) src/retrieval/build_dual_encoder.py --out_dir artifacts/retriever_bert
	$(PY) src/retrieval/infer_retrieve.py --rebuild_index

plan:
	$(PY) src/cli/demo.py plan --q "$(Q)"

exec:
	$(PY) src/cli/demo.py exec --q "$(Q)"

test:
	pytest -q


