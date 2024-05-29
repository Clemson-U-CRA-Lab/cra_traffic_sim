uint64 serial # serial id from virtual traffic simulation

int32 num_SVs_x # Number of traffic vehicles in the simulation
int32[12] virtual_vehicle_id # ID of traffic vehicles

float64[12] S_v_s # Traffic vehicles position in longitudinal direction
float64[12] S_v_l # Traffic vehicles position in lateral direction
float64[12] S_v_sv # Traffic vehicles speed in longitudinal direction
float64[12] S_v_acc # Traffic vehicles longitudinal acceleration
float64[12] S_v_yaw # Traffic vehicles yaw angle with respect to s direction
float64[12] S_v_omega # Traffic vehicles yaw rate with respect to s direction
bool[12] S_v_brake_status # Traffic vehicles braking status