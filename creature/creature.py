import numpy as np
import copy
from dataclasses import dataclass
from enum import Enum
from xml.dom.minidom import getDOMImplementation
from creature import genome, motor

@dataclass
class CreatureLink:
    name:str 
    gene_dict:dict
    parent_name:str = "None"
    recur:str = 1
    
    def __repr__(self) -> str:
        return f"URDF Link\nName\t: {self.name}\nParent\t: {self.parent_name}\nRecur\t: {self.recur}\n"

class Creature:
    __link_shapes = ("box", "cylinder", "sphere")
    __joint_types = ("revolute", "continuous")
    __joint_axes  = ("1 0 0", "0 1 0", "0 0 1")
    __counter = 0
    
    def __init__(self, gene_count, gene_spec = genome.GeneSpec.get_gene_spec()):
        self.dna = genome.Genome.init_random_genome(gene_count, len(gene_spec))
        self.spec = gene_spec
        self.start_position = (0, 0, 0)
        self.last_position = (0, 0, 0)
        self.motors = None
        self.__flat_links = None
        self.__expanded_links = None

    def get_flat_links(self):
        if self.__flat_links == None:
            g_dicts = genome.Genome.genome_to_dict(self.dna, self.spec)
            self.__flat_links = Creature.genome_to_links(g_dicts)
        return self.__flat_links

    def get_expanded_links(self):
        if self.__expanded_links == None:
            self.__expanded_links = Creature.expand_links(self.get_flat_links())
        self.__expanded_links[0].recur = 1
        return self.__expanded_links

    def get_robot_xml(self):
        return Creature.creature_to_xml(self.get_expanded_links())

    def write_robot_xml(self, path):
        if self.__expanded_links == None:
            self.__expanded_links = Creature.expand_links(self.get_flat_links())
        with open(path, "w") as f:
            xml_str = self.get_robot_xml().toprettyxml()
            f.write(xml_str)
        return None
    
    def get_motors(self):
        if self.motors == None:
            self.get_expanded_links()
            motors = []
            for i, link in enumerate(self.__expanded_links):
                if i == 0: continue
                motors.append(motor.Motor(  link.gene_dict["control_waveform"],
                                            link.gene_dict["control_amp"],
                                            link.gene_dict["control_freq"]))
            self.motors = motors
        return self.motors

    def reset_start_position(self, start_position):
        self.start_position = start_position
        return self.start_position

    def update_position(self, new_position):
        self.last_position = new_position
        return self.last_position

    def get_distance(self):
        return np.linalg.norm(np.asarray(self.last_position) - np.asarray(self.start_position))

    def update_dna(self, new_dna):
        assert len(self.spec) == new_dna.shape[-1]
        self.dna = new_dna
        self.start_position = (0, 0, 0)
        self.last_position = (0, 0, 0)
        self.get_flat_links()
        self.get_expanded_links()
        self.get_motors()

    @staticmethod
    def genome_to_links(genome_dicts):
        link_names = ["Link_" + str(i) for i in range(len(genome_dicts))]
        flat_links = []
        for i, gene_dict in enumerate(genome_dicts):
            if i == 0:
                parent_name, recur = "None", 1
            else:
                parent_name, recur = link_names[i-1], int(np.ceil(gene_dict["link_recurrence"]))
            flat_links.append(CreatureLink(link_names[i], gene_dict, parent_name, recur))
        return flat_links
    
    @staticmethod
    def expand_links(flat_links):        
        assert flat_links[0].recur == 1
        exp_links = Creature.__expand_links_recursive(flat_links[0], flat_links[1:])
        Creature.__counter = 0
        return exp_links

    @staticmethod
    def __expand_links_recursive(parent, child_links, child_id = "0"):
        p_copy = copy.copy(parent)
        p_copy.name += ("_" + str(child_id) + "_" + str(Creature.__counter))
        Creature.__counter += 1
        exp_links = [p_copy]
        if len(child_links) > 0:
            for i in range(child_links[0].recur):
                child = Creature.__expand_links_recursive(child_links[0], child_links[1:], i)
                c_copy = copy.copy(child)
                c_copy[0].parent_name = p_copy.name
                exp_links.extend(c_copy)
        return exp_links

    # @staticmethod
    # def __expand_links_recursive(parent, child_links):
    #     p_copy = copy.copy(parent)
    #     p_copy.name += ("_" + str(Creature.__counter))
    #     Creature.__counter += 1
    #     exp_links = [p_copy]
    #     if len(child_links) > 0:
    #         for i in range(child_links[0].recur):
    #             child = Creature.__expand_links_recursive(child_links[0], child_links[1:])
    #             c_copy = copy.copy(child)
    #             c_copy[0].parent_name = p_copy.name
    #             exp_links.extend(c_copy)
    #     return exp_links

    @staticmethod
    def link_to_xml(name, parent_name, gene_dict, adom, sib_ind = None):
        link_shape = Creature.__link_shapes[ gene_dict["link_shape"] ]
        joint_type = Creature.__joint_types[ gene_dict["joint_type"] ]
        joint_axis = Creature.__joint_axes[ gene_dict["joint_axis_xyz"] ]
        
        if sib_ind == None:
            try:
                sib_ind = int(name.split("_")[2]) # ex: Link_1_2_4
            except:
                sib_ind = 0

        shape_tag = adom.createElement(link_shape)
        if link_shape == "box":
            link_size = " ".join([str(gene_dict["link_length_1"]), str(gene_dict["link_length_2"]), str(gene_dict["link_length_3"])])
            shape_tag.setAttribute("size", str(link_size))
            link_volume    = gene_dict["link_length_1"] * gene_dict["link_length_2"] *  gene_dict["link_length_3"]
        elif link_shape =="cylinder":
            link_length = np.mean([gene_dict["link_length_1"], gene_dict["link_length_2"], gene_dict["link_length_3"]])
            link_radius = gene_dict["link_radius"]
            shape_tag.setAttribute("radius", str(link_radius))
            shape_tag.setAttribute("length", str(link_length))
            link_volume = np.pi * (gene_dict["link_length_1"] ** 2) * np.mean([
                gene_dict["link_length_1"],
                gene_dict["link_length_2"],
                gene_dict["link_length_3"]
            ])
        else:
            link_radius = gene_dict["link_radius"]
            shape_tag.setAttribute("radius", str(link_radius))
            link_volume = 4 / 3 * np.pi * (gene_dict["link_radius"] ** 3)

        link_mass = link_volume * gene_dict["link_mass_density"]

        # ----- LINK TAG -----
        mass_tag = adom.createElement("mass")
        mass_tag.setAttribute("value", str(link_mass))

        inertia_tag = adom.createElement("inertia")
        inertia_tag.setAttribute("ixx", "0.03")  
        inertia_tag.setAttribute("ixy", "0.03")  
        inertia_tag.setAttribute("ixz", "0.03") 
        inertia_tag.setAttribute("iyy", "0") 
        inertia_tag.setAttribute("iyz", "0") 
        inertia_tag.setAttribute("izz", "0")

        geometry_tag = adom.createElement("geometry")
        geometry_tag.appendChild(shape_tag)
        
        link_visual_tag = adom.createElement("visual")
        link_visual_tag.appendChild(copy.copy(geometry_tag))

        link_collision_tag = adom.createElement("collision")
        link_collision_tag.appendChild(geometry_tag)

        link_inertial_tag = adom.createElement("inertial")
        link_inertial_tag.appendChild(mass_tag)
        link_inertial_tag.appendChild(inertia_tag)

        link_tag = adom.createElement("link")
        link_tag.setAttribute("name", name)
        link_tag.appendChild(link_visual_tag)
        link_tag.appendChild(link_collision_tag)
        link_tag.appendChild(link_inertial_tag)
               
        # ----- JOINT TAG ----- 
        joint_parent_tag = adom.createElement("parent")
        joint_parent_tag.setAttribute("link", parent_name)

        joint_child_tag = adom.createElement("child")
        joint_child_tag.setAttribute("link", name)

        joint_origin_tag = adom.createElement("origin")
        joint_origin_tag.setAttribute("xyz", " ".join([
            str(gene_dict["joint_origin_xyz_1"] * sib_ind),
            str(gene_dict["joint_origin_xyz_2"]),
            str(gene_dict["joint_origin_xyz_3"])
        ]))
        joint_origin_tag.setAttribute("rpy", " ".join([
            str(gene_dict["joint_origin_rpy_1"]),
            str(gene_dict["joint_origin_rpy_2"]),
            str(gene_dict["joint_origin_rpy_3"])            
        ]))

        joint_axis_tag = adom.createElement("axis")
        joint_axis_tag.setAttribute("xyz", str(joint_axis))

        joint_limit_tag = adom.createElement("limit")
        joint_limit_tag.setAttribute("effort", "1")
        joint_limit_tag.setAttribute("upper", str(-np.pi))
        joint_limit_tag.setAttribute("lower", str(np.pi))
        joint_limit_tag.setAttribute("velocity", "1")

        joint_tag = adom.createElement("joint")
        joint_tag.setAttribute("name", "joint_" + name)
        joint_tag.setAttribute("type", joint_type)
        joint_tag.appendChild(joint_parent_tag)
        joint_tag.appendChild(joint_child_tag)
        joint_tag.appendChild(joint_axis_tag)
        joint_tag.appendChild(joint_origin_tag)
        joint_tag.appendChild(joint_limit_tag)

        return link_tag, joint_tag

    @staticmethod
    def creature_to_xml(links, robot_name = "robot"):
        adom = getDOMImplementation().createDocument(None, "start", None)
        robot_tag = adom.createElement("robot")
        robot_tag.setAttribute("name", robot_name)

        for i, link in enumerate(links):
            link_tag, joint_tag = Creature.link_to_xml(link.name, link.parent_name, link.gene_dict, adom)
            robot_tag.appendChild(link_tag)
            if i != 0:
                robot_tag.appendChild(joint_tag)

        return robot_tag        
        


if __name__ == "__main__":
    link = CreatureLink("hello", {}, np.empty(shape=1), "none")
    print(link)