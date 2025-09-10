from scipy.spatial.transform import Rotation as R

q = input("Enter quaternion components (w, x, y, z) separated by spaces: ")
q = [float(i) for i in q.split()]
r = R.from_quat(q)
rpy = r.as_euler('xyz')
print("Roll:", rpy[0])
print("Pitch:", rpy[1])
print("Yaw:", rpy[2])
