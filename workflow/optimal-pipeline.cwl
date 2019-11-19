class: Workflow
cwlVersion: v1.0
id: optimal_pipeline
label: optimal-pipeline
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
inputs:
  - id: coding
    type: string
    'sbg:x': 0
    'sbg:y': 107
  - id: xml_path
    type: string
    'sbg:x': 0
    'sbg:y': 0
  - id: abs_path
    type: string
    'sbg:x': 0
    'sbg:y': 214
  - id: drugbank_mapping
    type: string
    'sbg:x': 325.5736083984375
    'sbg:y': 46.5
outputs:
  - id: output
    outputSource:
      - annotate_druglabels/output
    type: File
    'sbg:x': 1014.1425170898438
    'sbg:y': 107
  - id: labels_cleaned
    outputSource:
      - clean_text/output
    type: File
    'sbg:x': 1020.1650390625
    'sbg:y': 284
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
    'sbg:x': 132.140625
    'sbg:y': 93
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
    'sbg:x': 325.5736083984375
    'sbg:y': 160.5
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
    'sbg:x': 565.3861083984375
    'sbg:y': 84
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
    'sbg:x': 784.881591796875
    'sbg:y': 100
requirements: []
