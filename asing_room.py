# Importar librerias
import re
from pulp import *
import pandas as pd
import numpy as np

# data frames
def leer_datos(path):

  def time_lapse(x):
    h = [int(i) for i in re.findall(r'\d*', x) if i.isdigit()]
    hours = [str(i) for i in range(h[0], h[1])]
    days = [day.upper() for day in re.findall(r'[aA-zZ]', x)]
    return' '.join([day+hour for day in days for hour in hours]) 

  grupos = pd.read_excel(open(path, 'rb'),
                sheet_name='Grupos')
  grupos['GRUPO'] = grupos.IDE + '-' +grupos.GR.apply(lambda x: str(x))
  grupos['GRUPO'] = grupos.GRUPO.apply(lambda x: x.replace(' ',''))
  grupos.set_index('GRUPO', inplace = True)
  grupos = grupos[['NOMBRE DE LA MATERIA', 'MAX',	'MEDIOS',	'HORARIO']]
  grupos.HORARIO = grupos.HORARIO.apply(lambda x: time_lapse(x))
  df = grupos.groupby(grupos.index).sum()
  dp = list(grupos[grupos.index.duplicated()].index)
  grupos = grupos[~grupos.index.duplicated(keep='first')]
  for i in dp:
    grupos['MAX'][i] = df['MAX'][i]

  aulas = pd.read_excel(open(path, 'rb'),
                sheet_name='Aulas')
  aulas.set_index('AULA', inplace = True)
  return grupos, aulas

def iniciar_diccionarios(aulas, grupos, dias):

  order_h = [dia+str(hora) for dia in dias for hora in range(6,23)]
  drai = {j:{k: '' for k in order_h} for j in aulas.index}
  prof = {dia:{grupo:'' for grupo in grupos.index} for dia in dias}

  resp = {'Tolerancia':{}, 'Medios':{},'asignados': {},'n_grupos': {},'alumnos': {},
          'sin_asignar': {}, 'n_grupos_sin': {}, 'alumnos_sin': {},
          'prom_sub_uso': {}, 'max_sub_uso': {}, 'min_sub_uso': {}
        }
  return drai, prof, resp

def grupo_horas(grupos):
  K = []
  [[K.append(x) for x in h] for h in [hr.split(' ') for hr in grupos.HORARIO]]
  K = list(set(K))
  L = []
  for i in grupos.index:
    l = []
    for k in K:
      if k in grupos.loc[i]['HORARIO']:
        l.append(1)
      else:
        l.append(0)
    L.append(l)
  return pd.DataFrame(L, index = grupos.index, columns=K)
  
def grupo_medios(grupos, aulas):
  L = []
  for i in grupos.index:
    l = []
    for j in aulas.index:
      if grupos.MEDIOS[i] == 1:
        if aulas['TIPO AULA'][j] == 'Aula Especial':
          l.append(1)
        else:
          l.append(0)
      else:
        l.append(1)        
    L.append(l)
  return pd.DataFrame(L, index = grupos.index, columns=aulas.index)

def grupo_capacidad(grupos, aulas, tolerancia = 0):
  L = []
  for i in grupos.index:
    l = []
    for j in aulas.index:
      diff = aulas['CAPACIDAD'][j] - grupos['MAX'][i] + tolerancia
      if diff < 0:
        l.append(0)
      else:
        l.append(1)        
    L.append(l)
  return pd.DataFrame(L, index = grupos.index, columns=aulas.index)


def diccionarios_dia(grupos):

  def lista_dias(grupos):
  
    l = []
    [[l.append(x) for x in h] for h in [hr.split(' ') for hr in grupos.HORARIO]]
    horas = list(set(l))
    dias = list(set([re.findall(r'[A-Z]', h)[0] for h in horas]))
    
    return dias, horas

  gr = {}
  hr = {}
  dias, horarios = lista_dias(grupos)
  for dia in dias:
    gr[dia] = sorted(list(grupos.index[grupos.HORARIO.apply(lambda x: dia in x)]))
    hr[dia] = sorted([hora for hora in horarios if dia in hora])
  
  return gr, hr

def define_parameters(aulas, dia, gr, hr):
  # Conjuntos
  I = gr[dia]
  J = set(aulas.index)
  K = hr[dia]

  return I, J, K

def solve_model(I, J, K, M, C, H, max_gr, s_piso, medios):

  prob = LpProblem('Aulas_grupos', LpMaximize)
  # Variables de desición
  x = pulp.LpVariable.dicts('asinacion', ((i, j) for i in I for j in J), cat='Binary')

  # Función objetivo
  prob += lpSum(x[i,j] * H[i,k]*max_gr[i] for i in I for j in J for k in K)

  # Restricciones
  # Se puede asignar maximo una grupo a una aula en la misma hora
  for j in J:
    for k in K:
      prob += lpSum(x[i,j] * H[i,k] for i in I) <= 1 

  # El grupo solo se asigna a la hora programada
  for i in I:
    for k in K:
      prob += lpSum(x[i,j] * H[i,k] for j in J) <= H[i,k]

  # Cada grupo se asigna a una sola aula
  for i in I:
      prob += lpSum(x[i,j] for j in J) <= 1

  if medios == 'obligatorio':
    # El aula asignada debe tener los medios requeridos
    for i in I:
      for j in J:
        prob += x[i,j] * M[i,j] == x[i,j]
  
  elif medios == 'flexible':
    # Prioridad en los medios
    for i in I:
      for j in J:
        prob += x[i,j] * M[i,j] == M[i,j]

  # El aula asignada debe tener la capacidad necesaria para el grupo
  for i in I:
    for j in J:
      prob += x[i,j] * C[i,j] == x[i,j]

  # El número total de alumnos en una hora debe ser inferior al número de sillas en el piso
  for k in K:
    prob += lpSum(x[i,j] * H[i,k]*max_gr[i] for i in I for j in J) <= s_piso
    
  ### Resolvemos e imprimimos el Status, si es Optimo, el problema tiene solución.
  prob.solve()
  print("Status:", LpStatus[prob.status])
  return prob

def extraer_resultados(dia, prob, grupos):
  l = [v.name.replace('_','-') for v in prob.variables() if v.varValue > 0]
  l = [re.findall(r'[A-Z]*\d*?-\d+', asing) for asing in l]

  salon = {v[0]: v[1] for v in l}
  horario = {i:[k for k in grupos.HORARIO[i].split() if dia in k] for i in salon.keys()}

  return salon, horario

def actualizar_dict(grupos, aulas, dia, salon, horario, drai, prof, resp, tolerancia,medios):

  for i in salon.keys():
    j = salon[i]
    prof[dia][i] = j
    for k in horario[i]:
      drai[j][k] = i  

  asignados = list(salon.keys())
  sin_asignar = [i for i in grupos.index if not i in asignados]

  resp['Tolerancia'][dia] = tolerancia
  resp['Medios'][dia] = medios
  resp['asignados'][dia] = asignados
  resp['n_grupos'][dia] = len(asignados)
  resp['alumnos'][dia] = grupos.MAX[asignados].sum()
  resp['sin_asignar'][dia] = sin_asignar
  resp['n_grupos_sin'][dia] = len(sin_asignar)
  resp['alumnos_sin'][dia] = grupos.MAX[sin_asignar].sum()
  resp['prom_sub_uso'][dia] = np.mean([aulas.CAPACIDAD[salon[i]] - grupos.MAX[i] for i in asignados])
  resp['max_sub_uso'][dia] = max([aulas.CAPACIDAD[salon[i]] - grupos.MAX[i] for i in asignados])
  resp['min_sub_uso'][dia] = min([aulas.CAPACIDAD[salon[i]] - grupos.MAX[i] for i in asignados])

  return drai, prof, resp

def run_model(path, tolerancia, medios='obligatorio', dias = ['L', 'M', 'W', 'J', 'V', 'S']):
  grupos, aulas = leer_datos(path)
  # Generar los diccionarios de salida
  drai, prof, resp = iniciar_diccionarios(aulas, grupos, dias)

  # Precalculando parametros
  Mij = grupo_medios(grupos, aulas) # Si el salón  j  tienes los medios que requiere el grupo  i
  Cij = grupo_capacidad(grupos, aulas)# Si el salón  j  tiene la capacidad que requiere el grupo  i
  Hik = grupo_horas(grupos) # Si el grupo i se debe dictar a la hora k

  # Parámetros como diccionario
  M = {(row, col) : Mij.loc[row,col] for row in Mij.index for col in Mij.columns}
  C = {(row, col) : Cij.loc[row,col] for row in Cij.index for col in Cij.columns}
  H = {(row, col) : Hik.loc[row,col] for row in Hik.index for col in Hik.columns}
  max_gr = {row: grupos.loc[row,'MAX'] for row in grupos.index}
  s_piso = aulas.CAPACIDAD.sum()
  gr, hr = diccionarios_dia(grupos)

  # Solucionar le modelo para cada día
  for dia in dias:
    if dia in gr.keys():
      # Diccionarios de subconjuntos
      I, J, K = define_parameters(aulas, dia, gr, hr)
      prob = solve_model(I, J, K, M, C, H, max_gr, s_piso, medios)
      salon, horario = extraer_resultados(dia, prob, grupos)
      drai, prof, resp = actualizar_dict(grupos, aulas, dia, salon, horario, drai, prof, resp, tolerancia,medios)
  return pd.DataFrame.from_dict(drai), pd.DataFrame.from_dict(prof), resp
