import numpy as np

templ = open('Radiative_template.txt', 'r')
cmd = templ.read()
templ.close()

file = open("Radiative_cmd.txt", 'w')

text = cmd.format(end_angle=90.0, angle_gap=5.0, is_in_spin='FALSE', spin_axis = [0.0, 1.0, 0.0], spin_pos = 18)

file.write(text)

file.close