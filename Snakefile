# Snakefile

rule all:
    input:
        "results/output.txt"

rule create_input:
    output:
        "data/input.txt"
    shell:
        "echo 'Hello, World!' > {output}"

rule process_data:
    input:
        "data/input.txt"
    output:
        "results/output.txt"
    shell:
        "cat {input} | tr 'a-z' 'A-Z' > {output}"
