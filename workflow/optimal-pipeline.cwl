class: Workflow
cwlVersion: v1.0
id: optimal_pipeline
label: optimal-pipeline
inputs:
  - id: coding
    type: string
  - id: xml_path
    type: string
  - id: abs_path
    type: string
  - id: drugbank_mapping
    type: string
outputs:
  - id: output
    outputSource:
      - annotate_druglabels/output
    type: File
steps:
  - id: extract_xml
    in:
      - id: coding
        source: coding
      - id: xml_path
        source: xml_path
      - id: abs_path
        source: abs_path
    out:
      - id: output
    run: steps/extract-xml.cwl
    label: extract-xml
  - id: clean_text
    in:
      - id: xml_product
        source: extract_xml/output
      - id: abs_path
        source: abs_path
    out:
      - id: output
    run: steps/clean-text.cwl
    label: clean-text
  - id: map2drugbank
    in:
      - id: drugbank_mapping
        source: drugbank_mapping
      - id: xml_product_cleaned
        source: clean_text/output
      - id: abs_path
        source: abs_path
    out:
      - id: output
    run: steps/map2drugbank.cwl
    label: map2DrugBank
  - id: annotate_druglabels
    in:
      - id: xml_product_dbid
        source: map2drugbank/output
      - id: abs_path
        source: abs_path
    out:
      - id: output
    run: steps/annotate-druglabels.cwl
    label: annotate-druglabels
requirements: []
