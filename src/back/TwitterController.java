package back;


import java.util.List;

import twitter4j.Query;
import twitter4j.QueryResult;
import twitter4j.Status;
import twitter4j.Twitter;
import twitter4j.TwitterException;
import twitter4j.TwitterFactory;
import twitter4j.conf.ConfigurationBuilder;


public class TwitterController {
	
	public TwitterController(){}
	
	public static QueryResult searchTweetsWith(String stringToSearch,int day, int month, int year, int finalday, int finalmonth, int finalyear)
	{
	    ConfigurationBuilder cb = new ConfigurationBuilder();
	    cb.setDebugEnabled(true)	//BOTAR AQUI OS VALORES
	          .setOAuthConsumerKey()
	          .setOAuthConsumerSecret()
	          .setOAuthAccessToken()
	          .setOAuthAccessTokenSecret();
	    TwitterFactory tf = new TwitterFactory(cb.build());
	    Twitter twitter = tf.getInstance();
	    List<Status> p;
	        try {
	            Query query = new Query(stringToSearch);
	            query.setLang("en");
	            query.setSince(year+"-"+month+"-"+day);
	            query.setUntil(finalyear+"-"+finalmonth+"-"+finalday);

	            QueryResult result;
	            result = twitter.search(query);
	            List<Status> tweets = result.getTweets();
	            int num = 0;
	            for (Status tweet : tweets) {
	                if(!tweet.getText().contains("RT") && !tweet.getText().contains("&&"))
	                {
	                	System.out.println(tweet.getText());
	                }
	            }

	            return result;
	        } catch (TwitterException te) {
	            te.printStackTrace();
	            return null;
	        }
	}
}
