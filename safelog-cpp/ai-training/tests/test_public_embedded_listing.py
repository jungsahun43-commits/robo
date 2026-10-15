import unittest
from scripts.inspect_mpdd2_complete_metadata import PublicEntries


class PublicListingTests(unittest.TestCase):
    def test_reads_names_and_ignores_scripts_and_remote_nonfile_links(self):
        parser=PublicEntries()
        parser.feed('<script>raise Exception("not executed")</script><div id="entry-abcdefghij"><a href="https://drive.google.com/file/d/abcdefghij/view?x=1"><div>A&amp;B.png</div></a></div><a href="https://other.example/file/d/other/view">ignored</a>')
        self.assertEqual(parser.result(),[{'id':'abcdefghij','name':'A&B.png','kind':'file'}])
    def test_mismatched_entry_and_link_is_rejected(self):
        parser=PublicEntries();parser.feed('<div id="entry-abcdefghij"><a href="https://other.example/">image.png</a></div>')
        with self.assertRaises(AssertionError):parser.result()


if __name__=='__main__':unittest.main()
