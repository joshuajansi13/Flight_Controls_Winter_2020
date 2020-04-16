import numpy as np
from message_types.msg_waypoints import msg_waypoints
from chap11.dubins_parameters import dubins_parameters

class planRRTDubins():
    def __init__(self):
        self.waypoints = msg_waypoints()
        self.segmentLength = 550  # standard length of path segments
        self.waypoints.type = 'dubins'
        self.dubins_path = dubins_parameters()
        self.dubins_path.radius = np.inf
        self.tree_courses = np.array([0.0])

    def planPath(self, wpp_start, wpp_end, map, R):

        self.dubins_path.radius = R

        # desired down position is down position of end node
        pd = wpp_end.item(2)

        # specify start and end nodes from wpp_start and wpp_end
        # format: N, E, D, cost, parentIndex, connectsToGoalFlag,
        start_node = np.array([wpp_start.item(0), wpp_start.item(1), pd, 0, 0, 0])
        end_node = np.array([wpp_end.item(0), wpp_end.item(1), pd, 0, 0, 0])

        # establish tree starting with the start node
        tree = start_node.reshape(1, -1)

        # check to see if start_node connects directly to end_node
        if ((np.linalg.norm(start_node[0:3] - end_node[0:3]) < self.segmentLength ) and not self.collision(start_node, end_node, map)):
            self.waypoints.ned = end_node[0:3]
        else:
            numPaths = 0
            while numPaths < 3:
                tree, flag = self.extendTree(tree, end_node, self.segmentLength, map, pd)
                numPaths = numPaths + flag


        # find path with minimum cost to end_node
        path = self.findMinimumPath(tree, end_node)
        waypoints_smoothed = self.smoothPath(path, map)
        self.waypoints.ned = waypoints_smoothed[:, :3].T
        self.waypoints.airspeed = np.ones(len(waypoints_smoothed)) * 25.0
        self.waypoints.course[0, 0] = 0.0
        for j in range(len(waypoints_smoothed) - 1):
            self.waypoints.course[0, j + 1] = np.arctan2(
                (self.waypoints.ned[1, j + 1] - self.waypoints.ned[1, j]),
                (self.waypoints.ned[0, j + 1] - self.waypoints.ned[0, j]))
        self.waypoints.num_waypoints = len(waypoints_smoothed)
        return self.waypoints

    def generateRandomNode(self, map, pd, chi):
        pn = np.random.uniform(0, map.city_width)
        pe = np.random.uniform(0, map.city_width)
        return np.array([pn, pe, pd])

    def collision(self, start_node, end_node, map):
        collided = False
        buffer = 50.  # make sure that we are not close to any obstacle
        pts = self.pointsAlongPath(start_node, end_node)
        for i in range(len(pts)):
            col_n = np.abs(map.building_north - pts[i, 0]) < (
                        map.building_width/2.0 + buffer)
            col_e = np.abs(map.building_east - pts[i, 1]) < (
                        map.building_width/2.0 + buffer)
            arg_n = np.where(col_n)[0]
            arg_e = np.where(col_e)[0]
            # checks to see if collided
            if len(arg_n) == 0 and len(arg_e) == 0:
                continue
            if map.building_height[arg_n, arg_e] >= -pts[i, 2]:
                collided = True
        return collided

    def pointsAlongPath(self, start_node, end_node):  #, Del):
        N = 100  # points along path
        d_theta = 0.1  # approx distance between arc points
        points_arc1 = self.pointsAlongArc(self.dubins_path.p_s,
                                          self.dubins_path.r1,
                                          self.dubins_path.center_s,
                                          self.dubins_path.dir_s, d_theta)
        points_line = self.pointsAlongLine(self.dubins_path.r1, self.dubins_path.r2, N)
        points_arc2 = self.pointsAlongArc(self.dubins_path.r2,
                                          self.dubins_path.p_e,
                                          self.dubins_path.center_e,
                                          self.dubins_path.dir_e, d_theta)
        points = np.vstack((points_arc1, points_line, points_arc2))
        return points

    def pointsAlongLine(self, p_s, p_e, N):
        # create vector pointing from start to end node
        q = (p_e - p_s) / np.linalg.norm(p_e - p_s)
        # find distance between each point along the path
        dist = np.linalg.norm(p_e - p_s) / N
        points = p_s.reshape(1, 3)
        for i in range(N):
            points = np.vstack((points, points[i, :] + q.T * dist))
        return points

    def pointsAlongArc(self, p_s, p_e, c, dir, d_th):
        p1 = (p_s - c).reshape(3, 1)
        p2 = (p_e - c).reshape(3, 1)
        p_i = p1
        d_theta = dir * d_th
        R = np.array([[np.cos(d_theta), -np.sin(d_theta), 0],
                      [np.sin(d_theta), np.cos(d_theta), 0],
                      [0, 0, 1]])
        cost = 0
        thresh = p2.T @ (R @ p2)
        points = p_s.T
        while cost < thresh:
            p_i = R @ p_i
            points = np.vstack((points, (p_i + c).T))
            cost = p_i.T @ p2
        points = np.vstack((points, p_e.T))
        return points

    def downAtNE(self, map, n, e):
        return

    def extendTree(self, tree, end_node, segmentLength, map, pd):
        valid_addition = False
        while not valid_addition:
            p = self.generateRandomNode(map, pd, 0)
            n_dist = (p.item(0) - tree[:, 0]) ** 2 + (
                        p.item(1) - tree[:, 1]) ** 2 + (
                                 p.item(2) - tree[:, 2]) ** 2
            parent = np.argmin(n_dist)
            n_closest = tree[parent, :3]
            q = (p - n_closest) / np.linalg.norm(p - n_closest)
            v_star = n_closest + q * segmentLength
            chi_s = self.tree_courses[parent]
            chi_e = np.arctan2((v_star.item(1) - n_closest.item(1)),
                               (v_star.item(0) - n_closest.item(0)))
            self.dubins_path.update(n_closest[0:3].reshape(3, 1), chi_s,
                                    v_star[0:3].reshape(3, 1), chi_e,
                                    self.dubins_path.radius)
            if self.collision(n_closest, v_star, map):
                continue
            else:
                cost = self.dubins_path.length + tree[parent, 3]
                n_new = np.append(v_star, [parent, cost, 0])
                tree = np.vstack((tree, n_new))
                self.tree_courses = np.append(self.tree_courses, chi_e)
                valid_addition = True
        if np.linalg.norm(end_node[0:3] - v_star) < segmentLength:
            flag = 1
            tree[-1, 5] = 1
        else:
            flag = 0
        return tree, flag

    def findMinimumPath(self, tree, end_node):
        term_pts = tree[np.where(tree[:, 5] == 1)]  # find all completion nodes
        path = np.array([term_pts[np.argmin(term_pts[:, 4])]])  # find minimum cost completion node
        finish = False
        while not finish:
            parent = path[0, 3]
            path = np.vstack((tree[int(parent)], path))
            if parent == 0:
                finish = True
                path = np.vstack((path, end_node))
        return path

    def smoothPath(self, path, map):
        w_s = np.array([path[0, :]])
        i = 0
        j = 1
        k = 0
        while j < len(path) - 1:
            chi_s = self.tree_courses[int(path[i, 3])]
            chi_e = np.arctan2((path[j+1, 1] - path[i, 1]), (path[j+1, 0] - path[i, 0]))
            self.dubins_path.update(path[i, 0:3].reshape(3, 1), chi_s,
                                    path[j + 1, 0:3].reshape(3, 1), chi_e,
                                    self.dubins_path.radius)
            if self.collision(path[i, :], path[j+1, :], map):
                w_s = np.vstack([w_s, path[j, :]])
                k += 1
                self.waypoints.course[0, k] = np.arctan2(
                    (w_s[k, 1] - w_s[k - 1, 1]), (w_s[k, 0] - w_s[k - 1, 0]))
                i = j
            j += 1
        self.waypoints.course[0, 0] = self.waypoints.course[0, 1]
        w_s = np.vstack([w_s, path[-1, :]])
        return w_s
