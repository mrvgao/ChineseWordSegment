"""
Deploy this application into internet.

RESTFul API.
"""

import tornado.ioloop
from summary.complete_sentenes_correlations import get_clean_top_sentences
import json
import logging
import tornado.web
from tornado.options import define, options, parse_command_line


class SummaryHandler(tornado.web.RequestHandler):
    def post(self):
        TITLE, CONTENT = 'title', 'content'
        get_argument = lambda key: self.request.body_arguments[key][0].decode('utf-8')
        title = get_argument(TITLE)
        content = get_argument(CONTENT)

        try:
            _summary, _suitability = get_clean_top_sentences(content, title)
            summary = {'summary': _summary, 'confidence': _suitability}
        except Exception as e:
            print(e)
            summary = {'summary': title + ':' + content, 'confidence': -1}
        self.write(summary)

def main():
    define("debug", default=False, help="run in debug mode")
    define("port", default=2334, help="run server on given port", type=int)
    parse_command_line()
    application = tornado.web.Application(
        [
            (r"/summary/", SummaryHandler),
        ],
        debug=True,
        xsrf_cookies=False,
    )
    application.listen(options.port)
    logging.info("Server running on port %d", options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()


