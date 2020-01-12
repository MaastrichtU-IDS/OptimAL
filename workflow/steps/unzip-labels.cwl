class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: unzip_labels
baseCommand:
  - python3
inputs:
  - id: input
    type: string
  - id: output
    type: string
  - id: abs_path
    type: string
outputs:
  - id: unzipped_dir
    type: Directory
    outputBinding:
      glob: $(inputs.output)
label: unzip-labels
arguments:
  - $(inputs.abs_path)src/obtainXML.py
  - '-d'
  - $(inputs.input)
  - '-o'
  - $(inputs.output)
requirements:
  - class: InlineJavascriptRequirement
