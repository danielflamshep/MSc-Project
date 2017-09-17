''' Use To access server with data'''
import paramiko


class DataConnector:
    def __init__(self, pswd, LAT=44.75, LON=-80.3125, LVL=None, version=2,
                 years=['2013','2014'], months=['05','06','07']):
        if LVL is None:
            self.levels = [str(x) for x in range(15)]
        else: self.level = LVL
        self.fname = ''.join(years)+''.join(months)+\
                     'LAT'+str(LAT)+'LON'+str(LON)+'LVL'+ \
                     self.levels[0] + '-' + self.levels[-1] + 'vers' + str(version)+'.npy'
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect('animus1.atmosp.physics.utoronto.ca', 22, username='dflamshe', password=pswd)
        print('connected to animus1')

    def transfer(self, remote='/home/dflamshe/data_processing.py',
                       local='E:/physicsMsc/research/code/data_processing.py'):
        sftp = self.client.open_sftp()
        sftp.put(local, remote)
        print('Transferred data processing')

    def run(self, cmd='python data_processing.py'):
        print('Building Data')
        self.stdin, self.stdout, self.stderr = self.client.exec_command(cmd, get_pty=True)
        self.stdin.close()
        for line in self.stdout.read().splitlines():
            print(line)

    def download(self):
        print('Transferring Data home')
        sftp = self.client.open_sftp()
        server_location = '/home/dflamshe/' + self.fname
        local = 'E:/physicsMsc/research/code/data/' + self.fname
        sftp.get(server_location, local)


if __name__ == '__main__':
    pswd = ""
    d=DataConnector(pswd)
    d.download()
