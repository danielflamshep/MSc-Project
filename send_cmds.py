import paramiko
import select

LAT=44.75;LON=-80.3125;LVL=1;version=1
years=['2013','2014']
months=['05','06','07']

file_name = ''.join(years)+''.join(months)+'LAT'+str(LAT)+'LON'+str(LON)+'LVL'+str(LVL)+str(version)+'.npy'
s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect('animus1.atmosp.physics.utoronto.ca', 22, username='dflamshe', password='zinho102')
print('Building Data')
stdin, stdout, stderr = s.exec_command('python data_processing.py', get_pty=True)
stdin.close()
for line in stdout.read().splitlines():
    print(line)

print('Transferring Data home')
sftp=s.open_sftp()
server_location = '/home/dflamshe/'+file_name
local = 'E:/physicsMsc/research/code/'+file_name
sftp.get(server_location, local)

print('Transfer Complete')