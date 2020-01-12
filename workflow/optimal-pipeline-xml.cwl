class: Workflow
cwlVersion: v1.0
id: optimal_pipeline
label: optimal-pipeline
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
inputs:
  - id: coding
    type: 'string[]'
    'sbg:x': -257.91571044921875
    'sbg:y': 192.51539611816406
  - id: abs_path
    type: string
    'sbg:x': -242.37277221679688
    'sbg:y': 353.3030700683594
  - id: drugbank_mapping
    type: string
    'sbg:x': -264.2220458984375
    'sbg:y': -110
  - id: api_key
    type: string
    'sbg:x': -258.5850830078125
    'sbg:y': -241.3938446044922
  - id: xml_path
    type: Directory
    'sbg:x': -271.93145751953125
    'sbg:y': 61.290611267089844
outputs:
  - id: annotated_with_doid
    outputSource:
      - annotate_druglabels/output
    type: File
    'sbg:x': 1039.222412109375
    'sbg:y': 103.11963653564453
  - id: cleaned_data
    outputSource:
      - clean_text/output
    type: File
    'sbg:x': 1034.700927734375
    'sbg:y': 377.3329162597656
  - id: labels_withDBID
    outputSource:
      - map2drugbank/output
    type: File
    'sbg:x': 1002.2223510742188
    'sbg:y': -120.88036346435547
steps:
  - id: extract_xml
    in:
      - id: coding
        source:
          - coding
      - id: xml_path
        source: xml_path
      - id: abs_path
        source: abs_path
    out:
      - id: output
    run: steps/extract-xml.cwl
    label: extract-xml
    'sbg:x': 191.941650390625
    'sbg:y': 190.73094177246094
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
    'sbg:x': 378.58953857421875
    'sbg:y': 248.26510620117188
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
    'sbg:x': 537.8374633789062
    'sbg:y': -90.61640930175781
  - id: annotate_druglabels
    in:
      - id: xml_product_dbid
        source: map2drugbank/output
      - id: abs_path
        source: abs_path
      - id: api_key
        source: api_key
    out:
      - id: output
    run: steps/annotate-druglabels.cwl
    label: annotate-druglabels
    'sbg:x': 784.881591796875
    'sbg:y': 100
requirements: []