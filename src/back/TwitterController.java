package back;


import java.util.List;
import java.util.Vector;

import twitter4j.Query;
import twitter4j.QueryResult;
import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.conf.ConfigurationBuilder;


public class TwitterController {
	
	public TwitterController(){}
	
	public static Vector<String> searchTweetsWith(String stringToSearch,String day,String month,String finalday,String finalmonth)
	{
	    ConfigurationBuilder cb = new ConfigurationBuilder();
	    cb.setDebugEnabled(true)
	          .setOAuthConsumerKey("Z6PXBblm7JLiCSBMGy4TMrG9z")
	          .setOAuthConsumerSecret("xYyO8DZQjBydMaWY53SXaJPy1owXgNMyD5RgHD8gCfbaQHd0oR")
	          .setOAuthAccessToken("4277442394-ajzWomJerOij6DKfFSxbi5Gy62tflz81Ct3KbyW")
	          .setOAuthAccessTokenSecret("xp8RQ2Sqc9YRmKl2PG5XspNaPBJxPAnblo6rkQR9QIy2W");
	    TwitterFactory tf = new TwitterFactory(cb.build());
	    Twitter twitter = tf.getInstance();
	    List<Status> p;
	        try {
	            Query query = new Query(stringToSearch);
	            query.setCount(40);
	            query.resultType(Query.MIXED);
	            query.setLang("en");
	           // query.setSince("2016-"+month+"-"+day);
	            query.setUntil("2016-"+finalmonth+"-"+finalday);

	            System.out.println("cuck");
	            QueryResult result;
	            result = twitter.search(query);
	            List<Status> tweets = result.getTweets();
	            int num = 0;
	            Vector<String> stringVector = new Vector<String>();
	            for (Status tweet : tweets) {
	            	System.out.println(tweet.getText());
	                if(!tweet.getText().contains("RT") && !tweet.getText().contains("http"))
	                {
	                	
	                	stringVector.add(tweet.getText());
	                }
	                if(stringVector.size()>23+2){
	                	return stringVector;
	                }
	            }

	            return stringVector;
	        } catch (TwitterException te) {
	            te.printStackTrace();
	            return null;
	        }
	}
	
}
