# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""gene_algorithm"""
import random
import os

from collections import namedtuple
import heapq

import numpy as np

from mindflow.utils import print_log

from .divide import divide_with_error


class BurgersGeneAlgorithm:
    """gene algorithm for burgers"""

    def __init__(self, config, case_name):
        super(BurgersGeneAlgorithm, self).__init__()
        self.ga_config = config["ga"]
        self.case_name = case_name
        self.max_iter = self.ga_config["max_iter"]
        self.partial_prob = self.ga_config["partial_prob"]
        self.genes_prob = self.ga_config["genes_prob"]
        self.cross_rate = self.ga_config["cross_rate"]
        self.mutate_rate = self.ga_config["mutate_rate"]
        self.delete_rate = self.ga_config["delete_rate"]
        self.add_rate = self.ga_config["add_rate"]
        self.pop_size = self.ga_config["pop_size"]
        self.n_generations = self.ga_config["n_generations"]
        self.meta_config = config["meta_data"]
        self.nx = self.meta_config["nx"]
        self.nt = self.meta_config["nt"]
        self.delete_num = self.ga_config["delete_num"]
        self.x = np.linspace(
            self.meta_config["x_min"], self.meta_config["x_max"], self.nx)
        self.t = np.linspace(
            self.meta_config["t_min"], self.meta_config["t_max"], self.nt)
        self.dx = self.x[1]-self.x[0]
        self.dt = self.t[1]-self.t[0]
        self.epi = self.ga_config["epi"]
        self.left_term = self.ga_config["left_term"]
        self.total_delete = (self.nx-self.delete_num)*(self.nt-self.delete_num)
        self.meta_path = os.path.join(
            self.meta_config["meta_data_save_path"], f"{case_name}_theta-ga.npy")
        self.r = np.load(self.meta_path)
        self.u = self.r[:, 0].reshape(self.r.shape[0], 1)
        self.u_x = self.r[:, 1].reshape(self.r.shape[0], 1)
        self.u_xx = self.r[:, 2].reshape(self.r.shape[0], 1)
        self.u_xxx = self.r[:, 3].reshape(self.r.shape[0], 1)
        self.u_t = self.r[:, 4].reshape(self.r.shape[0], 1)
        self.u_tt = self.r[:, 5].reshape(self.r.shape[0], 1)

    @staticmethod
    def delete_boundary(*data):
        u, nx, nt = data
        un = u.reshape(nx, nt)
        un_del = un[5:nx-5, 5:nt-5]
        return un_del.reshape((nx-10)*(nt-10), 1)

    @staticmethod
    def random_diff_module():
        diff_t = 0
        diff_x = random.randint(0, 3)
        genes_module = [diff_t, diff_x]
        return genes_module

    def finite_diff_x(self, un, dx, d):
        """cal finite diff x"""
        u = un.T
        nt, nx = u.shape
        ux = np.zeros([nt, nx])
        if d == 1:
            ux[:, nx - 1] = divide_with_error(
                (1.0 * u[:, nx - 1] - 2 * u[:, nx - 2] + divide_with_error(u[:, nx - 3], 2)), dx)
            ux[:, 0] = divide_with_error(
                (-1.5 * u[:, 0] + 2 * u[:, 1] - divide_with_error(u[:, 2], 2)), dx)
            ux[:, nx - 1] = divide_with_error(
                (1.0 * u[:, nx - 1] - 2 * u[:, nx - 2] + divide_with_error(u[:, nx - 3], 2)), dx)

            result = ux.T

        elif d == 2:
            ux[:, 1:nx - 1] = divide_with_error((u[:, 2:nx]-2 *
                                                 u[:, 1:nx-1]+u[:, 0:nx-2]), dx ** 2)
            ux[:, 0] = divide_with_error(
                (2 * u[:, 0]-5 * u[:, 1] + 4 * u[:, 2] - u[:, 3]), dx**2)
            ux[:, nx - 1] = divide_with_error((2 * u[:, nx-1]-5*u[:,
                                                                  nx-2] + 4 * u[:, nx-3] - u[:, nx-4]), dx**2)
            result = ux.T

        elif d == 3:
            ux[:, 2:nx - 2] = divide_with_error((divide_with_error(u[:, 4:nx], 2) - u[:, 3:nx-1] +
                                                 u[:, 1:nx-3] - divide_with_error(u[:, 0:nx-4], 2)), dx**3)
            ux[:, 0] = divide_with_error((-2.5 * u[:, 0] + 9 * u[:, 1] - 12 *
                                          u[:, 2] + 7 * u[:, 3]-1.5 * u[:, 4]), dx**3)
            ux[:, 1] = divide_with_error((-2.5 * u[:, 1] + 9 * u[:, 2]-12 *
                                          u[:, 3] + 7 * u[:, 4] - 1.5 * u[:, 5]), dx**3)
            ux[:, nx - 1] = divide_with_error((2.5 * u[:, nx-1] - 9 * u[:, nx-2] + 12 *
                                               u[:, nx-3] - 7 * u[:, nx-4] + 1.5 * u[:, nx-5]), dx**3)
            ux[:, nx - 2] = divide_with_error((2.5 * u[:, nx-2] - 9 * u[:, nx-3] + 12 *
                                               u[:, nx-4] - 7 * u[:, nx-5] + 1.5 * u[:, nx-6]), dx**3)
            result = ux.T

        else:
            result = self.finite_diff_x(self.finite_diff_x(u, dx, 3), dx, d-3)

        return result

    def finite_diff_t(self, un, dt, d):
        """cal finite_diff t"""
        u = un
        nx, nt = u.shape
        ux = np.zeros([nx, nt])
        if d == 1:
            ux[:, 1:nt - 1] = divide_with_error(
                (u[:, 2:nt] - u[:, 0:nt - 2]), (2 * dt))
            ux[:, 0] = divide_with_error(
                (-1.5 * u[:, 0] + 2 * u[:, 1] - divide_with_error(u[:, 2], 2)), dt)
            ux[:, nt-1] = divide_with_error((1.0 * u[:, nt - 1] - 2 *
                                             u[:, nt - 2] + divide_with_error(u[:, nt - 3], 2)), dt)
            result = ux

        elif d == 2:
            ux[:, 1:nt - 1] = divide_with_error((u[:, 2:nt] - 2 *
                                                 u[:, 1:nt-1] + u[:, 0:nt-2]), dt ** 2)
            ux[:, 0] = divide_with_error((2 * u[:, 0] - 5 * u[:, 1] +
                                          4 * u[:, 2] - u[:, 3]), dt**2)
            ux[:, nt - 1] = divide_with_error((2 * u[:, nt-1] - 5 *
                                               u[:, nt-2] + 4 * u[:, nt-3] - u[:, nt-4]), dt**2)
            result = ux

        elif d == 3:
            ux[:, 2:nt - 2] = divide_with_error((divide_with_error(u[:, 4:nt], 2) - u[:, 3:nt-1] +
                                                 u[:, 1:nt-3] - divide_with_error(u[:, 0:nt-4], 2)), dt**3)
            ux[:, 0] = divide_with_error((-2.5 * u[:, 0] + 9 * u[:, 1] - 12 *
                                          u[:, 2] + 7 * u[:, 3]-1.5 * u[:, 4]), dt**3)
            ux[:, 1] = divide_with_error((-2.5 * u[:, 1] + 9 * u[:, 2]-12 *
                                          u[:, 3] + 7 * u[:, 4] - 1.5 * u[:, 5]), dt**3)
            ux[:, nt - 1] = divide_with_error((2.5 * u[:, nt-1] - 9 * u[:, nt-2] + 12 *
                                               u[:, nt-3] - 7 * u[:, nt-4] + 1.5 * u[:, nt-5]), dt**3)
            ux[:, nt - 2] = divide_with_error((2.5 * u[:, nt-2] - 9 * u[:, nt-3] + 12 *
                                               u[:, nt-4] - 7 * u[:, nt-5] + 1.5 * u[:, nt-6]), dt**3)
            result = ux

        else:
            result = self.finite_diff_t(self.finite_diff_t(u, dt, 3), dt, d-3)

        return result

    def random_module(self):
        """get random module"""
        genes_module = []
        genes_diff_module = self.random_diff_module()
        for _ in range(self.max_iter):
            a = random.randint(0, 2)
            genes_module.append(a)
            prob = random.uniform(0, 1)
            if prob > self.partial_prob:
                break
        return genes_module, genes_diff_module

    def random_gene(self):
        """get random gene"""
        genes = []
        gene_lefts = []
        for _ in range(self.max_iter):
            gene_random, gene_random_diff = self.random_module()
            genes.append(sorted(gene_random))
            gene_lefts.append((gene_random_diff))
            prob = random.uniform(0, 1)
            if prob > self.genes_prob:
                break
        return genes, gene_lefts

    def translate_dna(self, gene_data):
        """translate dna"""
        gene, gene_left, u, u_x, u_xx, u_xxx = gene_data
        gene_translate = np.ones([self.total_delete, 1])
        length_penalty_coef = 0
        for _, (gene_module, gene_left_module) in enumerate(zip(gene, gene_left)):
            length_penalty_coef += len(gene_module)
            module_out = np.ones([u.shape[0], u.shape[1]])
            variables = [u, u_x, u_xx, u_xxx]
            for i in gene_module:
                temp = variables[i]
                module_out *= temp
            un = module_out.reshape(self.nx, self.nt)
            if gene_left_module[1] > 0:
                un_x = self.finite_diff_x(un, self.dx, d=gene_left_module[1])
                un = un_x
            if gene_left_module[0] > 0:
                un_t = self.finite_diff_t(un, self.dt, d=gene_left_module[0])
                un = un_t
            un = self.delete_boundary(un, self.nx, self.nt)
            module_out = un.reshape([self.total_delete, 1])
            gene_translate = np.hstack((gene_translate, module_out))
        gene_translate = np.delete(gene_translate, [0], axis=1)
        return gene_translate, length_penalty_coef

    def get_fitness(self, *data):
        """get fitness of genes"""
        gene_translate, u_t, length_penalty_coef = data
        u_t_new = u_t.reshape([self.nx, self.nt])
        u_t = self.delete_boundary(
            u_t_new, self.nx, self.nt).reshape(self.total_delete, 1)
        lst = np.linalg.lstsq(gene_translate, u_t, rcond=None)
        coef = lst[0]
        res = u_t-np.dot(gene_translate, coef)
        mse_true_value = divide_with_error(
            np.sum(np.array(res) ** 2), self.total_delete)
        mse_value = mse_true_value+self.epi*length_penalty_coef
        return coef, mse_value, mse_true_value

    def cross_over(self, gene, gene_left, father, father_left):
        """cross over"""
        cross_prob = random.uniform(0, 1)
        child = father.copy()
        child_left = father_left.copy()
        total_gene = gene.copy()
        total_gene_left = gene_left.copy()
        if cross_prob < self.cross_rate:
            mother_index = random.randint(0, len(gene)-1)
            mother = total_gene[mother_index]
            mother_left = total_gene_left[mother_index]
            swap_index_mother = random.randint(0, len(mother)-1)
            swap_index_father = random.randint(0, len(father)-1)
            swap_mother = mother[swap_index_mother]
            swap_mother_left = mother_left[swap_index_mother]
            child[swap_index_father] = swap_mother
            child_left[swap_index_father] = swap_mother_left
        return child, child_left

    def mutate(self, total_child, total_child_left):
        """mutate"""
        new_child = total_child.copy()
        new_child_left = total_child_left.copy()
        for i, (child, child_left) in enumerate(zip(total_child, total_child_left)):
            mutate_prob = random.uniform(0, 1)
            child_copy = child.copy()
            child_left_copy = child_left.copy()
            if mutate_prob < self.add_rate:
                child_index = random.randint(0, len(child_copy)-1)
                mutate_select = child_copy[child_index]
                gene_index = random.randint(0, len(mutate_select)-1)
                gene_select = mutate_select[gene_index]
                new_gene_left = self.random_diff_module()
                child_left_copy[child_index] = new_gene_left
                if gene_select == 3:
                    new_gene = 2
                if gene_select == 2:
                    new_gene = 1
                if gene_select == 1:
                    new_gene = 0
                if gene_select == 0:
                    new_gene = 3
                    child_copy[child_index][gene_index] = new_gene
                new_child_left[i] = child_left_copy
                new_child[i] = child_copy
            child_dele = new_child[i].copy()
            child_dele_left = new_child_left[i].copy()
            if len(child_dele) > 1:
                dele_prob = random.uniform(0, 1)
                if dele_prob < self.delete_rate:
                    delete_index = random.randint(0, len(child_dele) - 1)
                    child_dele.pop(delete_index)
                    child_dele_left.pop(delete_index)
                    new_child[i] = child_dele
                    new_child_left[i] = child_dele_left
            add_prob = random.uniform(0, 1)
            if add_prob < self.mutate_rate:
                add_gene, add_gene_left = self.random_module()
                new_child[i].append(add_gene)
                new_child_left[i].append(add_gene_left)

        return new_child, new_child_left

    def select(self, total_child, total_left):
        """nature selection wrt pop's fitness"""
        fitness_list = []
        new_left = []
        new_child = []
        new_fitness = []
        num = 0
        for _, (child, child_left) in enumerate(zip(total_child, total_left)):
            GeneData = namedtuple(
                'GeneData', ['gene', 'gene_left', 'u', 'u_x', 'u_xx', 'u_xxx'])
            child_translate, length_penalty_coef = self.translate_dna(
                GeneData(child, child_left, self.u, self.u_x, self.u_xx, self.u_xxx))
            mse_value = self.get_fitness(child_translate, self.u_t,
                                         length_penalty_coef)[1]
            fitness_list.append(mse_value)
            num += 1
        re1 = list(map(fitness_list.index, heapq.nsmallest(
            self.pop_size, fitness_list)))
        for index in re1:
            new_child.append(total_child[index])
            new_left.append(total_left[index])
            new_fitness.append(fitness_list[index])
        return new_child, new_left, new_fitness

    def get_genes(self):
        """get genes"""
        total_genes = []
        total_gene_left = []
        for _ in range(self.pop_size):
            gene, gene_left = self.random_gene()
            total_gene_left.append(gene_left)
            total_genes.append(gene)
        return total_genes, total_gene_left

    def get_child(self, total_genes, total_gene_left):
        """get child"""
        total_child_de = []
        total_child_left_de = []
        for _, (child, child_left) in enumerate(zip(total_genes, total_gene_left)):
            new_child = []
            combine_child_total = []
            new_child_left = []
            for _, (tmp_child, tmp_child_left) in enumerate(zip(child, child_left)):
                combine_child = [tmp_child, tmp_child_left]
                if combine_child not in combine_child_total:
                    if self.left_term == 'u_t':
                        if combine_child != [[0], [1, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        else:
                            new_child.append([0])
                            new_child_left.append([0, 1])
                            combine_child_total.append(combine_child)
                elif self.left_term == 'u_tt':
                    if combine_child != [[0], [2, 0]]:
                        new_child.append(sorted(tmp_child))
                        new_child_left.append(tmp_child_left)
                        combine_child_total.append(combine_child)
                        total_child_de.append(new_child)
                        total_child_left_de.append(new_child_left)
                    else:
                        new_child.append([0])
                        new_child_left.append([0, 1])
                        combine_child_total.append(combine_child)
            if new_child:
                total_child_de.append(new_child)
                total_child_left_de.append(new_child_left)

        child_select, left_select = self.select(
            total_child_de, total_child_left_de)[0:2]
        return child_select, left_select

    def evolve(self, child_select, left_select):
        """evolve"""
        for _ in range(self.n_generations):
            best = child_select[0].copy()
            best_left = left_select[0]
            best_save = np.array(best)
            best_left_save = np.array(best_left)
            np.save(f"{self.case_name}_best_save", best_save)
            np.save(f"{self.case_name}_best_left_save", best_left_save)
            child_loser = child_select.copy()
            left_loser = left_select.copy()
            child_loser.pop(0)
            left_loser.pop(0)
            total_genes = child_loser
            total_gene_left = left_loser
            total_child = []
            total_child_left = []
            for _, (father, father_left) in enumerate(zip(total_genes, total_gene_left)):
                child, child_left = self.cross_over(
                    total_genes, total_gene_left, father, father_left)
                total_child.append(child)
                total_child_left.append(child_left)
            self.mutate(total_child, total_child_left)
            total_child_de = []
            total_child_left_de = []
            for _, (child, child_left) in enumerate(zip(total_child, total_child_left)):
                new_child = []
                combine_child_total = []
                new_child_left = []
                for _, (tmp_child, tmp_child_left) in enumerate(zip(child, child_left)):
                    combine_child = [sorted(tmp_child), tmp_child_left]
                    if combine_child not in combine_child_total:
                        if self.left_term == 'u_t' and combine_child != [[0], [1, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                    elif self.left_term == 'u_tt':
                        if combine_child != [[0], [2, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                            total_child_de.append(new_child)
                            total_child_left_de.append(new_child_left)
                if new_child:
                    total_child_de.append(new_child)
                    total_child_left_de.append(new_child_left)

            best_load = np.load(f"{self.case_name}_best_save.npy", allow_pickle=True)
            best_left_load = np.load(f"{self.case_name}_best_left_save.npy", allow_pickle=True)
            best = best_load.tolist()
            best_left = best_left_load.tolist()
            total_child_de.insert(0, best)
            total_child_left_de.insert(0, best_left)

            child_select, left_select, fitness_select = self.select(
                total_child_de, total_child_left_de)
            total_child = child_select.copy()
            total_child_left = left_select.copy()
            print_log("-----------------------------")
            print_log(f"The best one: {child_select[0]}")
            GeneData = namedtuple(
                'GeneData', ['gene', 'gene_left', 'u', 'u_x', 'u_xx', 'u_xxx'])
            gene_translate, length_penalty_coef = self.translate_dna(GeneData(
                child_select[0], left_select[0], self.u, self.u_x, self.u_xx, self.u_xxx))
            coef, mse_value = self.get_fitness(
                gene_translate, self.u_t, length_penalty_coef)[0:2]
            print_log(fitness_select)
            print_log(f"The best coef: {coef}")
            print_log(f"The best MSE: {mse_value}")
            print_log(f"left is: {left_select[0]}")
            print_log(fitness_select)


class CylinderFlowGeneAlgorithm(BurgersGeneAlgorithm):
    """gene algorithm for cylinder_flow"""

    def __init__(self, config, case_name):
        super().__init__(config, case_name)
        self.ny = self.meta_config["ny"]
        self.y = np.linspace(
            self.meta_config["y_min"], self.meta_config["y_max"], self.ny)
        self.dy = self.y[1]-self.y[0]
        self.total_delete = (self.nx-self.delete_num) * \
            (self.ny-self.delete_num)*(self.nt-self.delete_num)
        self.u = self.r[:, 0].reshape(self.r.shape[0], 1)
        self.u_t = self.r[:, 1].reshape(self.r.shape[0], 1)
        self.u_x = self.r[:, 2].reshape(self.r.shape[0], 1)
        self.u_y = self.r[:, 3].reshape(self.r.shape[0], 1)
        self.u_xx = self.r[:, 4].reshape(self.r.shape[0], 1)
        self.u_yy = self.r[:, 5].reshape(self.r.shape[0], 1)
        self.u_xxx = self.r[:, 6].reshape(self.r.shape[0], 1)
        self.u_yyy = self.r[:, 7].reshape(self.r.shape[0], 1)

        self.v = self.r[:, 8].reshape(self.r.shape[0], 1)
        self.v_t = self.r[:, 9].reshape(self.r.shape[0], 1)
        self.v_x = self.r[:, 10].reshape(self.r.shape[0], 1)
        self.v_y = self.r[:, 11].reshape(self.r.shape[0], 1)
        self.v_xx = self.r[:, 12].reshape(self.r.shape[0], 1)
        self.v_yy = self.r[:, 13].reshape(self.r.shape[0], 1)
        self.v_xxx = self.r[:, 14].reshape(self.r.shape[0], 1)
        self.v_yyy = self.r[:, 15].reshape(self.r.shape[0], 1)

        self.p = self.r[:, 16].reshape(self.r.shape[0], 1)
        self.p_t = self.r[:, 17].reshape(self.r.shape[0], 1)
        self.p_x = self.r[:, 18].reshape(self.r.shape[0], 1)
        self.p_y = self.r[:, 19].reshape(self.r.shape[0], 1)
        self.p_xx = self.r[:, 20].reshape(self.r.shape[0], 1)
        self.p_yy = self.r[:, 21].reshape(self.r.shape[0], 1)
        self.p_xxx = self.r[:, 22].reshape(self.r.shape[0], 1)
        self.p_yyy = self.r[:, 23].reshape(self.r.shape[0], 1)

        self.h_t = self.u_t
        self.meta_data = namedtuple('MetaData', ['gene', 'gene_left',
                                                 'u', 'u_x', 'u_xx', 'u_xxx',
                                                 'u_y', 'u_yy', 'u_yyy',
                                                 'v', 'v_x', 'v_xx', 'v_xxx',
                                                 'v_y', 'v_yy', 'v_yyy',
                                                 'p', 'p_x', 'p_xx', 'p_xxx',
                                                 'p_y', 'p_yy', 'p_yyy'])

    @staticmethod
    def delete_boundary(*data):
        """delete boundary"""
        u, nx, ny, nt = data
        un = u.reshape(nx, ny, nt)
        un_del = un[5:nx-5, 5:ny-5, 5:nt-5]
        return un_del.reshape((nx-10)*(ny-10)*(nt-10), 1)

    def change_left_v(self):
        """change subject to v"""
        self.h_t = self.v_t

    def change_left_p(self):
        """change subject to p"""
        self.h_t = self.p_t

    def finite_diff_x(self, un, dx, d):
        """
        Compute the x-derivative of a function u(x,y,t) using finite differences.

        Parameters:
        un (ndarray): The function to differentiate.
        dx (float): The step size to use for the finite differences.
        d (int): The order of the derivative to compute.

        Returns:
        The x-derivative of u at each point (x,y,t).
        """
        u = un
        nx, ny, nt = u.shape
        ux = np.zeros([nx, ny, nt])

        if d == 1:
            ux[1: nx - 1, :, :] = divide_with_error(
                (u[2:nx, :, :] - u[0:nx-2, :, :]), (2 * dx))
            ux[0, :, :] = divide_with_error(
                (-1.5 * u[0, :, :] + 2 * u[1, :, :] - divide_with_error(u[2, :, :], 2)), dx)
            ux[nx - 1, :, :] = divide_with_error((1.0 * u[nx-1, :, :] -
                                                  2 * u[nx-2, :, :] + divide_with_error(u[nx-3, :, :], 2)), dx)
            result = ux

        elif d == 2:
            ux[1: nx-1, :, :] = divide_with_error(
                (u[2:nx, :, :] - 2 * u[1:nx-1, :, :] + u[0:nx-2, :, :]), dx ** 2)
            ux[0, :, :] = divide_with_error(
                (2 * u[0, :, :] - 5 * u[1, :, :] + 4 * u[2, :, :] - u[3, :, :]), dx ** 2)
            ux[nx - 1, :, :] = divide_with_error(
                (2 * u[nx-1, :, :] - 5 * u[nx-2, :, :] + 4 * u[nx-3, :, :] - u[nx-4, :, :]), dx ** 2)
            result = ux

        elif d == 3:
            ux[2:nx-2, :, :] = divide_with_error(
                (divide_with_error(u[4:nx, :, :], 2) -
                 u[3:nx-1, :, :] + u[1:nx-3, :, :] -
                 divide_with_error(u[0:nx-4, :, :], 2)), dx**3)
            ux[0, :, :] = divide_with_error((-2.5 * u[0, :, :] + 9 * u[1, :, :] -
                                             12 * u[2, :, :] + 7 * u[3, :, :] -
                                             1.5 * u[4, :, :]), dx**3)
            ux[1, :, :] = divide_with_error((-2.5 * u[1, :, :] + 9 * u[2, :, :] -
                                             12 * u[3, :, :] + 7 * u[4, :, :] -
                                             1.5 * u[5, :, :]), dx**3)
            ux[nx-1, :, :] = divide_with_error((2.5 * u[nx-1, :, :] - 9 * u[nx-2, :, :] +
                                                12 * u[nx-3, :, :] - 7 * u[nx-4, :, :] +
                                                1.5 * u[nx-5, :, :]), dx**3)
            ux[nx-2, :, :] = divide_with_error((2.5 * u[nx-2, :, :] - 9 * u[nx-3, :, :] +
                                                12 * u[nx-4, :, :] - 7 * u[nx-5, :, :] +
                                                1.5 * u[nx-6, :, :]), dx**3)
            result = ux
        else:
            result = self.finite_diff_x(self.finite_diff_x(u, dx, 3), dx, d-3)
        return result

    def finite_diff_y(self, un, dy, d):
        """
        Compute the y-derivative of a function u(x,y,t) using finite differences.

        Parameters:
        un (ndarray): The function to differentiate.
        dy (float): The step size to use for the finite differences.
        d (int): The order of the derivative to compute.

        Returns:
        The y-derivative of u at each point (x,y,t).
        """
        u = un
        nx, ny, nt = u.shape
        uy = np.zeros([nx, ny, nt])
        if d == 1:
            uy[:, 1:ny-1, :] = divide_with_error(
                (u[:, 2:ny, :] - u[:, 0:ny-2, :]), (2 * dy))
            uy[:, 0, :] = divide_with_error(
                (divide_with_error(-3.0, 2) * u[:, 0, :] +
                 2 * u[:, 1, :] - divide_with_error(u[:, 2, :], 2)), dy)
            uy[:, ny-1, :] = divide_with_error(
                (divide_with_error(2.0, 2) * u[:, ny-1, :] -
                 2 * u[:, ny-2, :] + divide_with_error(u[:, ny-3, :], 2)), dy)
            result = uy

        elif d == 2:
            uy[:, 1:ny-1, :] = divide_with_error((u[:, 2:ny, :] - 2 *
                                                  u[:, 1:ny-1, :] + u[:, 0:ny-2, :]), dy**2)
            uy[:, 0, :] = divide_with_error(
                (2 * u[:, 0, :] - 5 * u[:, 1, :] + 4 * u[:, 2, :] - u[:, 3, :]), dy**2)
            uy[:, ny-1, :] = divide_with_error((2 * u[:, ny-1, :] - 5 * u[:, ny-2, :] +
                                                4 * u[:, ny-3, :] - u[:, ny-4, :]), dy**2)
            result = uy

        elif d == 3:
            uy[:, 2:ny-2, :] = divide_with_error((divide_with_error(u[:, 4:ny, :], 2) - u[:, 3:ny-1, :] +
                                                  u[:, 1:ny-3, :] - divide_with_error(u[:, 0:ny-4, :], 2)), dy**3)
            uy[:, 0, :] = divide_with_error((-2.5 * u[:, 0, :] + 9 * u[:, 1, :] -
                                             12 * u[:, 2, :] + 7 * u[:, 3, :] - 1.5 * u[:, 4, :]), dy**3)
            uy[:, 1, :] = divide_with_error((-2.5 * u[:, 1, :] + 9 * u[:, 2, :] -
                                             12 * u[:, 3, :] + 7 * u[:, 4, :] - 1.5 * u[:, 5, :]), dy**3)
            uy[:, ny-1, :] = divide_with_error((2.5 * u[:, ny-1, :] - 9 * u[:, ny-2, :] +
                                                12 * u[:, ny-3, :] - 7 * u[:, ny-4, :] + 1.5 * u[:, ny-5, :]), dy**3)
            uy[:, ny-2, :] = divide_with_error((2.5 * u[:, ny-2, :] - 9 * u[:, ny-3, :] +
                                                12 * u[:, ny-4, :] - 7 * u[:, ny-5, :] + 1.5 * u[:, ny-6, :]), dy**3)
            result = uy

        else:
            result = self.finite_diff_y(self.finite_diff_y(u, dy, 3), dy, d-3)
        return result

    def finite_diff_t(self, un, dt, d):
        """
        Compute the t-derivative of a function u(x,y,t) using finite differences.

        Parameters:
        un (ndarray): The function to differentiate.
        dt (float): The step size to use for the finite differences.
        d (int): The order of the derivative to compute.

        Returns:
        The t-derivative of u at each point (x,y,t).
        """
        u = un
        nx, ny, nt = u.shape
        ut = np.zeros([nx, ny, nt])
        if d == 1:
            ut[:, :, 1:nt-1] = divide_with_error(
                (u[:, :, 2:nt] - u[:, :, 0:nt-2]), (2 * dt))
            ut[:, :, 0] = divide_with_error(
                (-1.5 * u[:, :, 0] + 2 * u[:, :, 1] -
                 divide_with_error(u[:, :, 2], 2)), dt)
            ut[:, :, nt-1] = divide_with_error(
                (u[:, :, nt-1] - 2 * u[:, :, nt-2] +
                 divide_with_error(u[:, :, nt-3], 2)), dt)
            result = ut

        elif d == 2:
            ut[:, :, 1:nt-1] = divide_with_error(
                (u[:, :, 2:nt] - 2 * u[:, :, 1:nt-1] + u[:, :, 0:nt-2]), dt**2)
            ut[:, :, 0] = divide_with_error(
                (2 * u[:, :, 0] - 5 * u[:, :, 1] + 4 * u[:, :, 2] - u[:, :, 3]), dt**2)
            ut[:, :, nt-1] = divide_with_error(
                (2 * u[:, :, nt-1] - 5 * u[:, :, nt-2] +
                 4 * u[:, :, nt-3] - u[:, :, nt-4]), dt**2)
            result = ut

        elif d == 3:
            ut[:, :, 2:nt-2] = divide_with_error(
                (divide_with_error(u[:, :, 4:nt], 2) - u[:, :, 3:nt-1] +
                 u[:, :, 1:nt-3] - divide_with_error(u[:, :, 0:nt-4], 2)), dt**3)
            ut[:, :, 0] = divide_with_error(
                (-2.5 * u[:, :, 0] + 9 * u[:, :, 1] -
                 12 * u[:, :, 2] + 7 * u[:, :, 3] - 1.5 * u[:, :, 4]), dt**3)
            ut[:, :, 1] = divide_with_error(
                (-2.5 * u[:, :, 1] + 9 * u[:, :, 2] -
                 12 * u[:, :, 3] + 7 * u[:, :, 4] - 1.5 * u[:, :, 5]), dt**3)
            ut[:, :, nt-1] = divide_with_error(
                (2.5 * u[:, :, nt-1] - 9 * u[:, :, nt-2] +
                 12 * u[:, :, nt-3] - 7 * u[:, :, nt-4] + 1.5 * u[:, :, nt-5]), dt**3)
            ut[:, :, nt-2] = divide_with_error(
                (2.5 * u[:, :, nt-2] - 9 * u[:, :, nt-3] +
                 12 * u[:, :, nt-4] - 7 * u[:, :, nt-5] + 1.5 * u[:, :, nt-6]), dt**3)
            result = ut

        else:
            result = self.finite_diff_t(self.finite_diff_t(u, dt, 3), dt, d-3)
        return result

    def random_diff_module(self):
        diff_t = 0
        diff_x = random.randint(0, 3)
        diff_y = random.randint(0, 3)
        genes_module = [diff_t, diff_x, diff_y]
        return genes_module

    def random_module(self):
        genes_module = []
        genes_diff_module = self.random_diff_module()
        for _ in range(self.max_iter):
            a = random.randint(0, 20)
            genes_module.append(a)
            prob = random.uniform(0, 1)
            if prob > self.partial_prob:
                break
        return genes_module, genes_diff_module

    def translate_dna(self, gene_data):
        gene, gene_left, u, u_x, u_xx, u_xxx, \
            u_y, u_yy, u_yyy, \
            v, v_x, v_xx, v_xxx, \
            v_y, v_yy, v_yyy, \
            p, p_x, p_xx, p_xxx, \
            p_y, p_yy, p_yyy = gene_data
        gene_translate = np.ones([self.total_delete, 1])
        length_penalty_coef = 0
        for _, (gene_module, gene_left_module) in enumerate(zip(gene, gene_left)):
            length_penalty_coef += len(gene_module)
            module_out = np.ones([u.shape[0], u.shape[1]])
            variables = [u, u_x, u_xx, u_xxx, u_y, u_yy, u_yyy,
                         v, v_x, v_xx, v_xxx, v_y, v_yy, v_yyy,
                         p, p_x, p_xx, p_xxx, p_y, p_yy, p_yyy]
            for i in gene_module:
                temp = variables[i]
                module_out *= temp
            un = module_out.reshape(self.nx, self.ny, self.nt)
            if gene_left_module[1] > 0:
                un_x = self.finite_diff_x(un, self.dx, d=gene_left_module[1])
                un = un_x
            if gene_left_module[2] > 0:
                un_y = self.finite_diff_y(un, self.dy, d=gene_left_module[2])
                un = un_y
            if gene_left_module[0] > 0:
                un_t = self.finite_diff_t(un, self.dt, d=gene_left_module[0])
                un = un_t
            un = self.delete_boundary(un, self.nx, self.ny, self.nt)
            module_out = un.reshape([self.total_delete, 1])
            gene_translate = np.hstack((gene_translate, module_out))
        gene_translate = np.delete(gene_translate, [0], axis=1)
        return gene_translate, length_penalty_coef

    def get_fitness(self, *data):
        gene_translate, h_t, length_penalty_coef = data
        h_t_new = h_t.reshape([self.nx, self.ny, self.nt])
        h_t = self.delete_boundary(
            h_t_new, self.nx, self.ny, self.nt).reshape(self.total_delete, 1)
        lst = np.linalg.lstsq(gene_translate, h_t, rcond=None)
        coef = lst[0]
        res = h_t-np.dot(gene_translate, coef)
        mse_true_value = divide_with_error(
            np.sum(np.array(res) ** 2), self.total_delete)
        mse_value = mse_true_value + self.epi * length_penalty_coef
        return coef, mse_value, mse_true_value

    # reach mutate
    def mutate(self, total_child, total_child_left):
        new_child = total_child.copy()
        new_child_left = total_child_left.copy()
        for i, (child, child_left) in enumerate(zip(total_child, total_child_left)):
            mutate_prob = random.uniform(0, 1)
            child_copy = child.copy()
            child_left_copy = child_left.copy()
            if mutate_prob < self.add_rate:
                child_index = random.randint(0, len(child_copy)-1)
                mutate_select = child_copy[child_index]
                gene_index = random.randint(0, len(mutate_select)-1)
                gene_select = mutate_select[gene_index]
                new_gene_left = self.random_diff_module()
                child_left_copy[child_index] = new_gene_left
                if gene_select != 0:
                    new_gene = gene_select - 1
                else:
                    new_gene = 20
                    child_copy[child_index][gene_index] = new_gene
                new_child_left[i] = child_left_copy
                new_child[i] = child_copy
            child_dele = new_child[i].copy()
            child_dele_left = new_child_left[i].copy()
            if len(child_dele) > 1:
                dele_prob = random.uniform(0, 1)
                if dele_prob < self.delete_rate:
                    delete_index = random.randint(0, len(child_dele) - 1)
                    child_dele.pop(delete_index)
                    child_dele_left.pop(delete_index)
                    new_child[i] = child_dele
                    new_child_left[i] = child_dele_left
            add_prob = random.uniform(0, 1)
            if add_prob < self.mutate_rate:
                add_gene, add_gene_left = self.random_module()
                new_child[i].append(add_gene)
                new_child_left[i].append(add_gene_left)

        return new_child, new_child_left

    def select(self, total_child, total_left):    # nature selection wrt pop's fitness
        fitness_list = []
        new_left = []
        new_child = []
        new_fitness = []
        num = 0
        for _, (child, child_left) in enumerate(zip(total_child, total_left)):
            child_translate, length_penalty_coef = self.translate_dna(
                self.meta_data(child, child_left, self.u, self.u_x, self.u_xx, self.u_xxx,
                               self.u_y, self.u_yy, self.u_yyy,
                               self.v, self.v_x, self.v_xx, self.v_xxx,
                               self.v_y, self.v_yy, self.v_yyy,
                               self.p, self.p_x, self.p_xx, self.p_xxx,
                               self.p_y, self.p_yy, self.p_yyy))
            mse_value = self.get_fitness(child_translate, self.h_t,
                                         length_penalty_coef)[1]
            fitness_list.append(mse_value)
            num += 1
        re1 = list(map(fitness_list.index, heapq.nsmallest(
            self.pop_size, fitness_list)))
        for index in re1:
            new_child.append(total_child[index])
            new_left.append(total_left[index])
            new_fitness.append(fitness_list[index])
        return new_child, new_left, new_fitness

    def get_child(self, total_genes, total_gene_left):
        total_child_de = []
        total_child_left_de = []
        for _, (child, child_left) in enumerate(zip(total_genes, total_gene_left)):
            new_child = []
            combine_child_total = []
            new_child_left = []
            for _, (tmp_child, tmp_child_left) in enumerate(zip(child, child_left)):
                combine_child = [sorted(tmp_child), tmp_child_left]
                if combine_child not in combine_child_total and self.left_term == 'u_t':
                    if self.h_t.all() == self.u_t.all():
                        if combine_child != [[0], [1, 0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        else:
                            new_child.append([0])
                            new_child_left.append([0, 1, 0])
                            combine_child_total.append(combine_child)
                    elif self.h_t.all() == self.v_t.all():
                        if combine_child != [[7], [1, 0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        else:
                            new_child.append([7])
                            new_child_left.append([0, 1, 0])
                            combine_child_total.append(combine_child)
                    else:
                        if combine_child != [[14], [1, 0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        else:
                            new_child.append([14])
                            new_child_left.append([0, 1, 0])
                            combine_child_total.append(combine_child)
            if new_child:
                total_child_de.append(new_child)
                total_child_left_de.append(new_child_left)

        child_select, left_select = self.select(
            total_child_de, total_child_left_de)[0:2]
        return child_select, left_select

    def evolve(self, child_select, left_select):
        """evolve"""
        for _ in range(self.n_generations):
            best = child_select[0].copy()
            best_left = left_select[0]
            best_save = np.array(best)
            best_left_save = np.array(best_left)
            np.save(f"{self.case_name}_best_save", best_save)
            np.save(f"{self.case_name}_best_left_save", best_left_save)
            child_loser = child_select.copy()
            left_loser = left_select.copy()
            child_loser.pop(0)
            left_loser.pop(0)
            total_genes = child_loser
            total_gene_left = left_loser
            total_child = []
            total_child_left = []
            for i, (father, father_left) in enumerate(zip(total_child, total_child_left)):
                father = total_genes[i]
                father_left = total_gene_left[i]
                child, child_left = self.cross_over(
                    total_genes, total_gene_left, father, father_left)
                total_child.append(child)
                total_child_left.append(child_left)
            self.mutate(total_child, total_child_left)
            total_child_de = []
            total_child_left_de = []
            for _, (child, child_left) in enumerate(zip(total_child, total_child_left)):
                new_child = []
                combine_child_total = []
                new_child_left = []
                for _, (tmp_child, tmp_child_left) in enumerate(zip(child, child_left)):
                    combine_child = [sorted(tmp_child), tmp_child_left]
                    if combine_child not in combine_child_total and self.left_term == 'u_t':
                        if self.h_t.all() == self.u_t.all() and combine_child != [[0], [1, 0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        elif self.h_t.all() == self.v_t.all() and combine_child != [[7], [1, 0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        elif combine_child != [[14], [1, 0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        else:
                            continue

                if new_child:
                    total_child_de.append(new_child)
                    total_child_left_de.append(new_child_left)

            best_load = np.load(f"{self.case_name}_best_save.npy", allow_pickle=True)
            best_left_load = np.load(f"{self.case_name}_best_left_save.npy", allow_pickle=True)
            best = best_load.tolist()
            best_left = best_left_load.tolist()
            total_child_de.insert(0, best)
            total_child_left_de.insert(0, best_left)

            child_select, left_select, fitness_select = self.select(
                total_child_de, total_child_left_de)
            total_child = child_select.copy()
            total_child_left = left_select.copy()
            print_log("-----------------------------")
            print_log(f"The best one: {child_select[0]}")
            gene_translate, length_penalty_coef = self.translate_dna(
                self.meta_data(child_select[0], left_select[0], self.u, self.u_x, self.u_xx, self.u_xxx,
                               self.u_y, self.u_yy, self.u_yyy,
                               self.v, self.v_x, self.v_xx, self.v_xxx,
                               self.v_y, self.v_yy, self.v_yyy,
                               self.p, self.p_x, self.p_xx, self.p_xxx,
                               self.p_y, self.p_yy, self.p_yyy))
            coef, mse_value = self.get_fitness(
                gene_translate, self.h_t, length_penalty_coef)[0:2]
            print_log(fitness_select)
            print_log(f"The best coef: {coef}")
            print_log(f"The best MSE: {mse_value}")
            print_log(f"left is: {left_select[0]}")


class PeriodicHillGeneAlgorithm(BurgersGeneAlgorithm):
    """GeneAlgorithm for PeriodicHill"""

    def __init__(self, config, case_name):
        """init"""
        super().__init__(config, case_name)
        self.ga_config = config["ga"]
        self.max_iter = self.ga_config["max_iter"]
        self.partial_prob = self.ga_config["partial_prob"]
        self.genes_prob = self.ga_config["genes_prob"]
        self.cross_rate = self.ga_config["cross_rate"]
        self.mutate_rate = self.ga_config["mutate_rate"]
        self.delete_rate = self.ga_config["delete_rate"]
        self.add_rate = self.ga_config["add_rate"]
        self.pop_size = self.ga_config["pop_size"]
        self.n_generations = self.ga_config["n_generations"]
        self.meta_config = config["meta_data"]
        self.delete_num = self.ga_config["delete_num"]

        self.nx = self.meta_config["nx"]
        self.ny = self.meta_config["ny"]
        self.x = np.linspace(
            self.meta_config["x_min"], self.meta_config["x_max"], self.nx)
        self.y = np.linspace(
            self.meta_config["y_min"], self.meta_config["y_max"], self.ny)
        self.dx = self.x[1]-self.x[0]
        self.dy = self.y[1]-self.y[0]
        self.epi = self.ga_config["epi"]
        self.left_term = self.ga_config["left_term"]

        self.meta_path = os.path.join(
            self.meta_config["meta_data_save_path"], f"{case_name}_theta-ga.npy")

        self.total_delete = (self.nx-self.delete_num) * \
            (self.ny-self.delete_num)

        self.r = np.load(self.meta_path)
        self.u = self.r[:, 0].reshape(self.r.shape[0], 1)
        self.u_x = self.r[:, 1].reshape(self.r.shape[0], 1)
        self.u_y = self.r[:, 2].reshape(self.r.shape[0], 1)
        self.u_xx = self.r[:, 3].reshape(self.r.shape[0], 1)
        self.u_yy = self.r[:, 4].reshape(self.r.shape[0], 1)
        self.u_xxx = self.r[:, 5].reshape(self.r.shape[0], 1)
        self.u_yyy = self.r[:, 6].reshape(self.r.shape[0], 1)

        self.v = self.r[:, 7].reshape(self.r.shape[0], 1)
        self.v_x = self.r[:, 8].reshape(self.r.shape[0], 1)
        self.v_y = self.r[:, 9].reshape(self.r.shape[0], 1)
        self.v_xx = self.r[:, 10].reshape(self.r.shape[0], 1)
        self.v_yy = self.r[:, 11].reshape(self.r.shape[0], 1)
        self.v_xxx = self.r[:, 12].reshape(self.r.shape[0], 1)
        self.v_yyy = self.r[:, 13].reshape(self.r.shape[0], 1)

        self.p = self.r[:, 14].reshape(self.r.shape[0], 1)
        self.p_x = self.r[:, 15].reshape(self.r.shape[0], 1)
        self.p_y = self.r[:, 16].reshape(self.r.shape[0], 1)
        self.p_xx = self.r[:, 17].reshape(self.r.shape[0], 1)
        self.p_yy = self.r[:, 18].reshape(self.r.shape[0], 1)
        self.p_xxx = self.r[:, 19].reshape(self.r.shape[0], 1)
        self.p_yyy = self.r[:, 20].reshape(self.r.shape[0], 1)

        self.h_t = self.u

        self.meta_data = namedtuple('MetaData', ['gene', 'gene_left',
                                                 'u', 'u_x', 'u_xx', 'u_xxx',
                                                 'u_y', 'u_yy', 'u_yyy',
                                                 'v', 'v_x', 'v_xx', 'v_xxx',
                                                 'v_y', 'v_yy', 'v_yyy',
                                                 'p', 'p_x', 'p_xx', 'p_xxx',
                                                 'p_y', 'p_yy', 'p_yyy'])

    @staticmethod
    def delete_boundary(*data):
        """delete boundary"""
        u, nx, ny = data
        un = u.reshape(nx, ny)
        un_del = un[5:nx-5, 5:ny-5]
        return un_del.reshape((nx-10)*(ny-10), 1)

    def change_left_v(self):
        """change subject to v"""
        self.h_t = self.v

    def change_left_p(self):
        """change subject to p"""
        self.h_t = self.p

    def random_module(self):
        genes_module = []
        genes_diff_module = self.random_diff_module()
        for _ in range(self.max_iter):
            a = random.randint(0, 20)
            genes_module.append(a)
            prob = random.uniform(0, 1)
            if prob > self.partial_prob:
                break
        return genes_module, genes_diff_module

    def random_diff_module(self):
        """get random diff module"""
        diff_x = random.randint(0, 3)
        diff_y = random.randint(0, 3)
        genes_module = [diff_x, diff_y]
        return genes_module

    def translate_dna(self, gene_data):
        """translate dna"""
        gene, gene_left, u, u_x, u_xx, u_xxx, \
            u_y, u_yy, u_yyy, v, v_x, v_xx, v_xxx, \
            v_y, v_yy, v_yyy, p, p_x, p_xx, p_xxx, \
            p_y, p_yy, p_yyy = gene_data
        gene_translate = np.ones([self.total_delete, 1])
        length_penalty_coef = 0
        for _, (gene_module, gene_left_module) in enumerate(zip(gene, gene_left)):
            length_penalty_coef += len(gene_module)
            module_out = np.ones([u.shape[0], u.shape[1]])
            variables = [u, u_x, u_xx, u_xxx, u_y, u_yy, u_yyy,
                         v, v_x, v_xx, v_xxx, v_y, v_yy, v_yyy,
                         p, p_x, p_xx, p_xxx, p_y, p_yy, p_yyy]
            for i in gene_module:
                temp = variables[i]
                module_out *= temp
            un = module_out.reshape(self.nx, self.ny)
            if gene_left_module[0] > 0:
                un_x = self.finite_diff_x(un, self.dx, d=gene_left_module[0])
                un = un_x
            if gene_left_module[1] > 0:
                un_y = self.finite_diff_y(un, self.dy, d=gene_left_module[1])
                un = un_y
            un = self.delete_boundary(un, self.nx, self.ny)
            module_out = un.reshape([self.total_delete, 1])
            gene_translate = np.hstack((gene_translate, module_out))
        gene_translate = np.delete(gene_translate, [0], axis=1)
        return gene_translate, length_penalty_coef

    def finite_diff_x(self, un, dx, d):
        """
        Compute the x-derivative of a function u(x,y) using finite differences.

        Parameters:
        un (ndarray): The function to differentiate.
        dx (float): The step size to use for the finite differences.
        d (int): The order of the derivative to compute.

        Returns:
        The x-derivative of u at each point (x,y).
        """
        u = un
        nx, ny = u.shape
        ux = np.zeros([nx, ny])

        if d == 1:
            ux[1: nx - 1, :] = divide_with_error(
                (u[2:nx, :] - u[0:nx-2, :]), (2 * dx))
            ux[0, :] = divide_with_error(
                (-1.5 * u[0, :] + 2 * u[1, :] - divide_with_error(u[2, :], 2)), dx)
            ux[nx - 1, :] = divide_with_error(
                (1.5 * u[nx-1, :] - 2 * u[nx-2, :] + divide_with_error(u[nx-3, :], 2)), dx)
            result = ux

        elif d == 2:
            ux[1: nx-1, :] = divide_with_error(
                (u[2:nx, :] - 2 * u[1:nx-1, :] + u[0:nx-2, :]), dx ** 2)
            ux[0, :] = divide_with_error(
                (2 * u[0, :] - 5 * u[1, :] + 4 * u[2, :] - u[3, :]), dx ** 2)
            ux[nx - 1, :] = divide_with_error(
                (2 * u[nx-1, :] - 5 * u[nx-2, :] + 4 * u[nx-3, :] - u[nx-4, :]), dx ** 2)
            result = ux

        elif d == 3:
            ux[2:nx-2, :] = divide_with_error((divide_with_error(
                u[4:nx, :], 2) - u[3:nx-1, :] + u[1:nx-3, :] - divide_with_error(u[0:nx-4, :], 2)), dx**3)
            ux[0, :] = divide_with_error(
                (-2.5 * u[0, :] + 9 * u[1, :] - 12 * u[2, :] + 7 * u[3, :] - 1.5 * u[4, :]), dx**3)
            ux[1, :] = divide_with_error(
                (-2.5 * u[1, :] + 9 * u[2, :] - 12 * u[3, :] + 7 * u[4, :] - 1.5 * u[5, :]), dx**3)
            ux[nx-1, :] = divide_with_error((2.5 * u[nx-1, :] - 9 * u[nx-2, :] +
                                             12 * u[nx-3, :] - 7 * u[nx-4, :] + 1.5 * u[nx-5, :]), dx**3)
            ux[nx-2, :] = divide_with_error((2.5 * u[nx-2, :] - 9 * u[nx-3, :] +
                                             12 * u[nx-4, :] - 7 * u[nx-5, :] + 1.5 * u[nx-6, :]), dx**3)
            result = ux
        else:
            result = self.finite_diff_x(self.finite_diff_x(u, dx, 3), dx, d-3)
        return result

    def finite_diff_y(self, un, dy, d):
        """
        Compute the y-derivative of a function u(x,y) using finite differences.

        Parameters:
        un (ndarray): The function to differentiate.
        dy (float): The step size to use for the finite differences.
        d (int): The order of the derivative to compute.

        Returns:
        The y-derivative of u at each point (x,y).
        """
        u = un
        nx, ny = u.shape
        uy = np.zeros([nx, ny])

        if d == 1:
            uy[:, 1:ny - 1] = divide_with_error(
                (u[:, 2:ny] - u[:, 0:ny-2]), (2 * dy))
            uy[:, 0] = divide_with_error(
                (-1.5 * u[:, 0] + 2 * u[:, 1] - divide_with_error(u[:, 2], 2)), dy)
            uy[:, ny - 1] = divide_with_error(
                (1.5 * u[:, ny-1] - 2 * u[:, ny-2] + divide_with_error(u[:, ny-3], 2)), dy)
            result = uy

        elif d == 2:
            uy[:, 1:ny-1] = divide_with_error(
                (u[:, 2:ny] - 2 * u[:, 1:ny-1] + u[:, 0:ny-2]), dy ** 2)
            uy[:, 0] = divide_with_error(
                (2 * u[:, 0] - 5 * u[:, 1] + 4 * u[:, 2] - u[:, 3]), dy ** 2)
            uy[:, ny - 1] = divide_with_error(
                (2 * u[:, ny-1] - 5 * u[:, ny-2] + 4 * u[:, ny-3] - u[:, ny-4]), dy ** 2)
            result = uy

        elif d == 3:
            uy[:, 2:ny-2] = divide_with_error(
                (divide_with_error(u[:, 4:ny], 2) - u[:, 3:ny-1] + u[:, 1:ny-3] -
                 divide_with_error(u[:, 0:ny-4], 2)), dy**3)
            uy[:, 0] = divide_with_error(
                (-2.5 * u[:, 0] + 9 * u[:, 1] - 12 * u[:, 2] + 7 * u[:, 3] - 1.5 * u[:, 4]), dy**3)
            uy[:, 1] = divide_with_error(
                (-2.5 * u[:, 1] + 9 * u[:, 2] - 12 * u[:, 3] + 7 * u[:, 4] - 1.5 * u[:, 5]), dy**3)
            uy[:, ny-1] = divide_with_error(
                (2.5 * u[:, ny-1] - 9 * u[:, ny-2] + 12 * u[:, ny-3] - 7 * u[:, ny-4] + 1.5 * u[:, ny-5]), dy**3)
            uy[:, ny-2] = divide_with_error(
                (2.5 * u[:, ny-2] - 9 * u[:, ny-3] + 12 * u[:, ny-4] - 7 * u[:, ny-5] + 1.5 * u[:, ny-6]), dy**3)
            result = uy
        else:
            result = self.finite_diff_y(self.finite_diff_y(u, dy, 3), dy, d-3)
        return result

    def get_fitness(self, *data):
        """get fitness"""
        gene_translate, h_t, length_penalty_coef = data
        h_t_new = h_t.reshape([self.nx, self.ny])
        h_t = self.delete_boundary(
            h_t_new, self.nx, self.ny).reshape(self.total_delete, 1)
        lst = np.linalg.lstsq(gene_translate, h_t, rcond=None)
        coef = lst[0]
        res = h_t-np.dot(gene_translate, coef)
        mse_true_value = divide_with_error(
            np.sum(np.array(res) ** 2), self.total_delete)
        mse_value = mse_true_value+self.epi*length_penalty_coef
        return coef, mse_value, mse_true_value

    # reach mutate
    def mutate(self, total_child, total_child_left):
        new_child = total_child.copy()
        new_child_left = total_child_left.copy()
        for i, (child, child_left) in enumerate(zip(total_child, total_child_left)):
            mutate_prob = random.uniform(0, 1)
            child_copy = child.copy()
            child_left_copy = child_left.copy()
            if mutate_prob < self.add_rate:
                child_index = random.randint(0, len(child_copy)-1)
                mutate_select = child_copy[child_index]
                gene_index = random.randint(0, len(mutate_select)-1)
                gene_select = mutate_select[gene_index]
                new_gene_left = self.random_diff_module()
                child_left_copy[child_index] = new_gene_left
                if gene_select != 0:
                    new_gene = gene_select - 1
                else:
                    new_gene = 20
                    child_copy[child_index][gene_index] = new_gene
                new_child_left[i] = child_left_copy
                new_child[i] = child_copy
            child_dele = new_child[i].copy()
            child_dele_left = new_child_left[i].copy()
            if len(child_dele) > 1:
                dele_prob = random.uniform(0, 1)
                if dele_prob < self.delete_rate:
                    delete_index = random.randint(0, len(child_dele) - 1)
                    child_dele.pop(delete_index)
                    child_dele_left.pop(delete_index)
                    new_child[i] = child_dele
                    new_child_left[i] = child_dele_left
            add_prob = random.uniform(0, 1)
            if add_prob < self.mutate_rate:
                add_gene, add_gene_left = self.random_module()
                new_child[i].append(add_gene)
                new_child_left[i].append(add_gene_left)

        return new_child, new_child_left

    def select(self, total_child, total_left):    # nature selection wrt pop's fitness
        fitness_list = []
        new_left = []
        new_child = []
        new_fitness = []
        num = 0
        for _, (child, child_left) in enumerate(zip(total_child, total_left)):
            child_translate, length_penalty_coef = self.translate_dna(
                self.meta_data(child, child_left, self.u, self.u_x, self.u_xx, self.u_xxx,
                               self.u_y, self.u_yy, self.u_yyy,
                               self.v, self.v_x, self.v_xx, self.v_xxx,
                               self.v_y, self.v_yy, self.v_yyy,
                               self.p, self.p_x, self.p_xx, self.p_xxx,
                               self.p_y, self.p_yy, self.p_yyy))
            mse_value = self.get_fitness(child_translate, self.h_t,
                                         length_penalty_coef)[1]
            fitness_list.append(mse_value)
            num += 1
        re1 = list(map(fitness_list.index, heapq.nsmallest(
            self.pop_size, fitness_list)))
        for index in re1:
            new_child.append(total_child[index])
            new_left.append(total_left[index])
            new_fitness.append(fitness_list[index])
        return new_child, new_left, new_fitness

    def get_child(self, total_genes, total_gene_left):
        """get child"""
        total_child_de = []
        total_child_left_de = []
        for _, (child, child_left) in enumerate(zip(total_genes, total_gene_left)):
            new_child = []
            combine_child_total = []
            new_child_left = []
            for _, (tmp_child, tmp_child_left) in enumerate(zip(child, child_left)):
                combine_child = [sorted(tmp_child), tmp_child_left]
                if combine_child not in combine_child_total:
                    if self.left_term == 'u' and self.h_t.all() == self.u.all():
                        if combine_child != [[0], [0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        else:
                            new_child.append([0])
                            new_child_left.append([1, 0])
                            combine_child_total.append(combine_child)
                    elif self.left_term == 'u' and self.h_t.all() == self.v.all():
                        if combine_child != [[7], [0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        else:
                            new_child.append([7])
                            new_child_left.append([1, 0])
                            combine_child_total.append(combine_child)
                    elif self.left_term == 'u' and self.h_t.all() == self.p.all():
                        if combine_child != [[14], [0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        else:
                            new_child.append([14])
                            new_child_left.append([1, 0])
                            combine_child_total.append(combine_child)
            if new_child:
                total_child_de.append(new_child)
                total_child_left_de.append(new_child_left)

        child_select, left_select = self.select(
            total_child_de, total_child_left_de)[0:2]
        return child_select, left_select

    def evolve(self, child_select, left_select):
        """evolve"""
        for _ in range(self.n_generations):
            best = child_select[0].copy()
            best_left = left_select[0]
            best_save = np.array(best)
            best_left_save = np.array(best_left)
            np.save(f"{self.case_name}_best_save", best_save)
            np.save(f"{self.case_name}_best_left_save", best_left_save)
            child_loser = child_select.copy()
            left_loser = left_select.copy()
            child_loser.pop(0)
            left_loser.pop(0)
            total_genes = child_loser
            total_gene_left = left_loser
            total_child = []
            total_child_left = []
            for _, (father, father_left) in enumerate(zip(total_genes, total_gene_left)):
                child, child_left = self.cross_over(
                    total_genes, total_gene_left, father, father_left)
                total_child.append(child)
                total_child_left.append(child_left)
            self.mutate(total_child, total_child_left)
            total_child_de = []
            total_child_left_de = []
            for _, (child, child_left) in enumerate(zip(total_child, total_child_left)):
                new_child = []
                combine_child_total = []
                new_child_left = []
                for _, (tmp_child, tmp_child_left) in enumerate(zip(child, child_left)):
                    combine_child = [sorted(tmp_child), tmp_child_left]
                    if combine_child not in combine_child_total and self.left_term == 'u':
                        if self.h_t.all() == self.u.all() and combine_child != [[0], [0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        elif self.h_t.all() == self.v.all() and combine_child != [[7], [0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        elif self.h_t.all() == self.p.all() and combine_child != [[14], [0, 0]]:
                            new_child.append(sorted(tmp_child))
                            new_child_left.append(tmp_child_left)
                            combine_child_total.append(combine_child)
                        else:
                            continue
                if new_child:
                    total_child_de.append(new_child)
                    total_child_left_de.append(new_child_left)

            best_load = np.load(f"{self.case_name}_best_save.npy", allow_pickle=True)
            best_left_load = np.load(f"{self.case_name}_best_left_save.npy", allow_pickle=True)
            best = best_load.tolist()
            best_left = best_left_load.tolist()
            total_child_de.insert(0, best)
            total_child_left_de.insert(0, best_left)

            child_select, left_select, fitness_select = self.select(
                total_child_de, total_child_left_de)
            total_child = child_select.copy()
            total_child_left = left_select.copy()
            print_log("-----------------------------")
            print_log(f"The best one: {child_select[0]}")
            gene_translate, length_penalty_coef = self.translate_dna(
                self.meta_data(
                    child_select[0], left_select[0], self.u, self.u_x, self.u_xx, self.u_xxx,
                    self.u_y, self.u_yy, self.u_yyy,
                    self.v, self.v_x, self.v_xx, self.v_xxx,
                    self.v_y, self.v_yy, self.v_yyy,
                    self.p, self.p_x, self.p_xx, self.p_xxx,
                    self.p_y, self.p_yy, self.p_yyy))
            coef, mse_value = self.get_fitness(
                gene_translate, self.h_t, length_penalty_coef)[0:2]
            print_log(fitness_select)
            print_log(f"The best coef: {coef}")
            print_log(f"The best MSE: {mse_value}")
            print_log(f"left is: {left_select[0]}")
