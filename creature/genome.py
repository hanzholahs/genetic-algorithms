import numpy as np

class Genome:
    @staticmethod
    def init_random_gene(n):
        gene = np.random.uniform(size = n)
        return gene

    @staticmethod
    def init_random_genome(gene_count, gene_length): 
        genome = np.array([Genome.init_random_gene(gene_length) for _ in range(gene_count)])
        return genome

    @staticmethod
    def gene_to_dict(gene, spec):
        gene_dict = {}
        for key in spec.keys():
            gene_val = gene[spec[key]["index"]] * spec[key]["scale"]
            if spec[key]["type"] == "discrete":
                gene_val = int(np.ceil(gene_val))
            elif spec[key]["type"] == "categorical":
                gene_val = int(np.ceil(gene_val) - 1)
            gene_dict[key] = gene_val

        return gene_dict

    @staticmethod
    def genome_to_dict(genome, spec):
        genome_dicts = [Genome.gene_to_dict(gene, spec) for gene in genome]
        return genome_dicts

class GeneSpec:
    __spec = None

    @staticmethod
    def set_default_gene_spec():
        spec = {
            "link_shape":{"scale":3, "type":"categorical"},
            "link_length_1": {"scale":1},
            "link_length_2": {"scale":1},
            "link_length_3": {"scale":1},
            "link_radius": {"scale":1},
            "link_recurrence": {"scale":3, "type":"discrete"},
            "link_mass_density": {"scale":5},
            "joint_type": {"scale":2, "type":"categorical"},
            "joint_axis_xyz": {"scale":3, "type": "categorical"},
            "joint_origin_rpy_1":{"scale":np.pi * 2},
            "joint_origin_rpy_2":{"scale":np.pi * 2},
            "joint_origin_rpy_3":{"scale":np.pi * 2},
            "joint_origin_xyz_1":{"scale":1},
            "joint_origin_xyz_2":{"scale":1},
            "joint_origin_xyz_3":{"scale":1},
            "control_waveform":{"scale":2, "type":"categorical"},
            "control_amp":{"scale":0.25},
            "control_freq":{"scale":1}
        }
        ind = 0
        for key in spec.keys():
            if "type" not in spec[key].keys():
                spec[key]["type"] = "continuous"

            spec[key]["index"] = ind
            ind += 1

        GeneSpec.__spec = spec
        return GeneSpec.__spec

    @staticmethod
    def get_gene_spec():
        if GeneSpec.__spec == None:
            GeneSpec.set_default_gene_spec()
        return GeneSpec.__spec

    @staticmethod
    def set_gene_spec(spec):
        GeneSpec.valid_spec(spec)

        GeneSpec.__spec = spec
        return GeneSpec.__spec

    @staticmethod
    def valid_spec(spec):
        assert type(spec) == dict, f"Input is not a dictionary"

        index_occurences = []

        for key in spec.keys():
            assert type(spec[key]) == dict, f"Specification {key} is not a dictionary"
            
            assert "type" in spec[key], f"There is no key 'type' in the specification {spec[key]}"
            assert "scale" in spec[key], f"There is no key 'scale' in the specification {spec[key]}"
            assert "index" in spec[key], f"There is no key 'index' in the specification {spec[key]}"

            assert spec[key]["index"] not in index_occurences, f"Double index"
            assert spec[key]["type"] in ["discrete", "continuous", "categorical"], f"Invalid specification 'type' in the specification {spec[key]}"
            assert isinstance(spec[key]["scale"], float) or isinstance(spec[key]["scale"], int), "Specification 'scale' is not a 'float' or 'int'"

            if spec[key]["index"] not in index_occurences:
                index_occurences.append(spec[key]["index"])

        return True