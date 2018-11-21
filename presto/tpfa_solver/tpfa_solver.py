#!/usr/bin/env python3
import sys
import numpy as np
from pymoab import core, types, rng
from pymoab.tag import Tag
from math import ceil

# Variáveis globais
# num_elem_i -> Número de elementos ao longo do eixo i
# di -> Tamanho do elemento ao longo do eixo i
# dim -> Número de dimensões das adjacências dos elementos (ex.: numa malha 2D,
#        os elementos compartilham arestas que tem dim = 1)
num_elem_x = 10
num_elem_y = 3
num_elem_z = 1
dx = 2.0
dy = 5/3
dz = 1
dim = 2
num_elements = 30

# create_mesh_connectivity: Função para montagem da conectividade das elementos.
# Parâmetros:
#   - vertex_handles: vetor de EntityHandles representando vértices do MOAB.
#   - vertex_coords: coordenadas dos vértices
def create_mesh_connectivity(vertex_handles, vertex_coords):
    global num_elem_x, num_elem_y, num_elem_z, dx, dy, dz, num_elements

    x_inf, y_inf, z_inf = 0, 0, 0
    x_sup, y_sup, z_sup = dx, dy, dz
    m, n = 0, 0

    mesh_connectivity = np.zeros((num_elements, 8), dtype=np.uint64)

    while x_sup <= num_elem_x*dx:
        while y_sup <= num_elem_y*dy:
            while z_sup <= num_elem_z*dz:
                n = 0
                for i in range(len(vertex_handles)):
                    x, y, z = vertex_coords[3*i], vertex_coords[3*i+1], vertex_coords[3*i+2]
                    if (x <= x_sup and x >= x_inf) and (y <= y_sup and y >= y_inf) and (z <= z_sup and z >= z_inf):
                        mesh_connectivity[m][n] = vertex_handles[i]
                        n += 1
                z_inf = z_sup
                z_sup += dz
                if (n == 8):
                    m += 1
            y_inf = y_sup
            y_sup += dy
            z_inf = 0.0
            z_sup = dz
        x_inf = x_sup
        x_sup += dx
        y_inf = 0.0
        y_sup = dy

    return mesh_connectivity

def get_centroid_coords(v):
    global dx, dy, dz
    centroid_x = v[0] + (dx/2)
    centroid_y = v[1] + (dy/2)
    centroid_z = v[2] + (dz/2)
    return np.array([centroid_x, centroid_y, centroid_z])

def equiv_perm(k1, k2):
    return (2*k1*k2)/(k1 + k2)

def centroid_dist(c1, c2):
    return (c1[0] + c2[0])**2 + (c1[1] + c2[1])**2 + (c1[2] + c2[2])**2


def main():
    global num_elem_x, num_elem_y, num_elem_z, dx, dy, dz, dim, num_elements

    # Tratamento da entrada. O número de dimensões da malha é determinado a partir
    # da quantidade de argumentos.
    if len(sys.argv) == 7:
        num_elem_x = int(sys.argv[1])
        num_elem_y = int(sys.argv[3])
        num_elem_z = int(sys.argv[5])
        dx = float(sys.argv[2])
        dy = float(sys.argv[4])
        dz = float(sys.argv[6])
        dim = 2
        num_elements = num_elem_x*num_elem_y*num_elem_z
    else:
        print("Not enough arguments")
        return

    # Criando instância da classe Core que gerencias as operações na malha.
    mbcore = core.Core()

    # Inicializando o vetor de coordenadas dos vértices.
    vertex_coords = np.array([])
    for k in range(num_elem_z + 1):
        for j in range(num_elem_y + 1):
            for i in range(num_elem_x + 1):
                vertex_coords = np.append(vertex_coords, [i*dx, j*dy, k*dz])

    # O método create_vertices cria os handles associados a cada coordenada em vertex_coords
    vertex_handles = mbcore.create_vertices(vertex_coords)

    # Em mesh_connectivity são aramazenados os conjuntos de vértices que compõem um elemento,
    # ou seja, determina a conectividade dos vértices na malha.
    mesh_connectivity = create_mesh_connectivity(vertex_handles, vertex_coords)

    # De posse da conectividade da malha, criamos os elementos um a um. A troca de valores
    # nas duas primeiras linhas do laço são necessárias devido a forma como o MOAB interpreta
    # a malha e as adjacências dos elementos.
    elem_handles = rng.Range()
    for c in mesh_connectivity:
        c[2], c[3] = c[3], c[2]
        c[6], c[7] = c[7], c[6]
        temp = mbcore.create_element(types.MBHEX, c)
        elem_handles.insert(temp)

    # Incialização da matriz de conectividade. (Neste caso, a conectividade é em
    # relação aos elementos, ou seja, quais elementos são vizinhos.)
    connectivity = np.zeros((num_elements, num_elements), dtype=np.bool_)

    # Encontrando adjacências para preencher a matriz de conectividade
    adjacencies = [mbcore.get_adjacencies(e, dim, True) for e in elem_handles]

    # Para cada adjacência diferente, verifica-se se existem uma fronteira compartilhada.
    # Caso positivo, os dois elementos são vizinhos e isto é indicado em connectivity.
    i, j = 0, 0
    for a in adjacencies:
        for b in adjacencies:
            if b != a:
                intersection = rng.intersect(a, b)
                if not intersection.empty():
                    connectivity[i][j] = 1
                    connectivity[j][i] = 1
            j += 1
        j = 0
        i += 1

    # Determinando as coordenadas do centroide de cada elemento e aramazenando-as em tags.
    # Uma tag é um valor associado a cada elemento. Aqui, cada elemento possui duas tags: uma
    # que armazena o valor das coordenadas do centroide e outra que armazena a permeabilidade.
    centroid_tag = mbcore.tag_get_handle('centroid', 3, types.MB_TYPE_DOUBLE, types.MB_TAG_DENSE, True)
    permeability_tag = mbcore.tag_get_handle('permeability', 1, types.MB_TYPE_DOUBLE, types.MB_TAG_DENSE, True)
    for e in elem_handles:
        elem_vertex = mbcore.get_connectivity(e)
        centroid_coord = get_centroid_coords(mbcore.get_coords([elem_vertex[0]]))
        mbcore.tag_set_data(centroid_tag, e, centroid_coord)
        mbcore.tag_set_data(permeability_tag, e, np.array([1], dtype=np.float_))

    mbcore.write_file("tpfa_mesh.h5m")
    print("New h5m file created")

    # Montagem da matriz de coeficientes do sistema.
    coef = np.zeros((num_elements, num_elements), dtype=np.float_)
    for i in range(num_elements):
        for j in range(num_elements):
            # Se dois elementos são vizinhos e não se trata do mesmo elemento, então
            # são recuperados os valores das tags e calculado o valor do coeficiente.
            if connectivity[i][j] == True and i != j:
                e1_tags = mbcore.tag_get_tags_on_entity(elem_handles[i])
                e2_tags = mbcore.tag_get_tags_on_entity(elem_handles[j])
                e1_centroid = mbcore.tag_get_data(e1_tags[0], elem_handles[i], flat=True)
                print(e1_centroid);
                e2_centroid = mbcore.tag_get_data(e2_tags[0], elem_handles[j], flat=True)
                e1_perm = mbcore.tag_get_data(e1_tags[1], elem_handles[i], flat=True)[0]
                e2_perm = mbcore.tag_get_data(e2_tags[1], elem_handles[j], flat=True)[0]
                coef[i][j] = (-1)*equiv_perm(e1_perm, e2_perm)/centroid_dist(e1_centroid, e2_centroid)
        coef[i][i] = (-1)*coef[i].sum()

    for x in coef:
        print(x)


if __name__ == '__main__':
    main()
