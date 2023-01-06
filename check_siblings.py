from creature import creature
from creature import genome

spec = genome.GeneSpec.get_gene_spec()
gene = genome.Genome.init_random_gene(len(spec))
gene_dict = genome.Genome.gene_to_dict(gene, spec)

links = [
    creature.CreatureLink("A", parent_name = "None", gene_dict = gene_dict, recur = 1),
    creature.CreatureLink("B", parent_name = "A", gene_dict = gene_dict, recur = 1),
    creature.CreatureLink("C", parent_name = "B", gene_dict = gene_dict, recur = 3),
    creature.CreatureLink("D", parent_name = "C", gene_dict = gene_dict, recur = 1)
]

exp_links = creature.Creature.expand_links(links)

for link in exp_links:
    print(link)