per = (0,8)

p = {
    "Leading_Photons_dR":[per, 'Angular Distance between two leading Photons [rad]', 'Angular Distnace [rad]','LeadPhoton_eta','SubleadPhoton_eta','LeadPhoton_phi','SubleadPhoton_phi'],
    "Diphoton_fathbbjet_dR":[per, 'Angular Distance between the Diphoton System and the Fat Hbb Jet [rad]', 'Angular Distance','Diphoton_eta','fathbbjet_eta','Diphoton_phi','fathbbjet_phi'],
    "Lead_fathbbjet_dR":[per,'Angular Distance between the Lead Photon and the Fat Hbb Jet [rad]', 'Angular Distance','LeadPhoton_eta','fathbbjet_eta','LeadPhoton_phi','fathbbjet_phi'],
    "Sublead_fathbbjet_dRs":[per,'Angular Distance between the Sublead Photon and the Fat Hbb Jet [rad]', 'Angular Distance','SubleadPhoton_eta','fathbbjet_eta','SubleadPhoton_phi','fathbbjet_phi']
}
keys = list(p.keys())

values = list(p.values())



ranges= []
titles = []
xaxis = []
temp_vals =[]
vals =[]

for l in range(len(values)):

    ranges.append(values[l][0])
    titles.append(values[l][1])
    xaxis.append(values[l][2])
    temp_vals.append(values[l][3:])
    vals.append(temp_vals)
    temp_vals =[]





print(ranges)
print('=================================\n\n\n\n')

print(keys)
print('=================================\n\n\n\n')

print(titles)
print('=================================\n\n\n\n')

print(xaxis)
print('=================================\n\n\n\n')

print(vals)
print('=================================\n\n\n\n')







print(vals[0])
