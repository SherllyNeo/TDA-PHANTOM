class DisplayPersistenceDiagram:
    def __init__(self,point_cloud: np.ndarray, is_distance_matrixd: bool = False, dgms: np.ndarry = None, plot: str = "both"):
        self.point_cloud = point_cloud
        self.is_distance_matrix = is_distance_matrix
        self.dgms = dgms 
        self.plot = plor

class DisplaySignificancePersistenceDiagram:
    def __init__(self,results: dict = None, method: str = "all", plot: str = "both"):
        self.results = results
        self.method = method
        self.plot = plot

