from locust import HttpLocust, TaskSet, task

class UserBehavior(TaskSet):
    def on_start(self):
        """ on_start is called when a Locust start before any task is scheduled """
        self.index()

    def index(self):
        self.client.get("")

    @task(1)
    def post_img(self):
        filepath_1 = 'practice.jpg'
        filepath_2 = 'practice.jpg'
        self.client.post("upload", files={'filearg1': filepath_1, 'filearg2': filepath_2})

class WebsiteUser(HttpLocust):
    task_set = UserBehavior
    min_wait = 5000
    max_wait = 9000
