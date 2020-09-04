
  /* ---------------------- Variables --------------------------- */

  let isGalleryDisplayed = true;
  let dropDownActive = false;

  let project_filters = ["1", "2"];

  let social_navbar_text = [
    "#linkedin-navbar-text", 
    "#github-navbar-text", 
    "#medium-navbar-text", 
    "#publication-navbar-text", 
    "#twitter-navbar-text", 
    "#kaggle-navbar-text"
  ];

  let navbar_social_links = [
    "#linkedin-navbar-social-link",
    "#github-navbar-social-link",
    "#medium-navbar-social-link",
    "#publication-navbar-social-link",
    "#twitter-navbar-social-link",
    "#kaggle-navbar-social-link"
  ];

  // Toggle boolean for if mobile navbar is active
  $(function(){
    $("#navbar-mobile-btn").click(function(){
      dropDownActive = !dropDownActive;
      if (dropDownActive) {
        $("#gallery-container").addClass("visibility-hidden");
      } else {
        $("#gallery-container").removeClass("visibility-hidden");
      };
    });
  });

  /* ------------------- Document is Ready ---------------------- */

  $(document).ready(function () {

    /* ---------------------- DOM is Ready -------------------------- */

    // Large Devices: Add text to social media buttons and appropriate css styles
    if($(this).width() > 1400) {
      $("#linkedin-navbar-text").text("Linkedin ");
      $("#github-navbar-text").text("Github ");
      $("#medium-navbar-text").text("Medium ");
      $("#publication-navbar-text").text("Publication ");
      $("#twitter-navbar-text").text("Twitter ");
      $("#kaggle-navbar-text").text("Kaggle ");

      // Update class for social buttons with text
      $.each(navbar_social_links, function(index, value) {
        $(value).addClass("navbar-social-btn-width-with-text");
      });

    // Medium or smaller device:
    } else {

      // Remove text for social buttons
      $.each(social_navbar_text, function(index, value) {
        $(value).text("");
      });

      // Apply appropriate css styles
      $.each(navbar_social_links, function(index, value) {
        $(value).addClass("navbar-social-btn-width-with-no-text");
      });
    }

    /* ---------------------- Resizing ---------------------------- */

    // When resizing, check width and adjust accordingly
    $(function(){
      $(window).on('resize', function(){

        // For large devices
        if($(this).width() > 1400) {

          // Add labels to social media buttons
          $("#linkedin-navbar-text").text("Linkedin ");
          $("#github-navbar-text").text("Github ");
          $("#medium-navbar-text").text("Medium ");
          $("#publication-navbar-text").text("Publication ");
          $("#twitter-navbar-text").text("Twitter ");
          $("#kaggle-navbar-text").text("Kaggle ");

          // Apply css changes for social media buttons with text
          $.each(navbar_social_links, function(index, value){

            if ($(value).hasClass("navbar-social-btn-width-with-no-text")) {
              $(value).removeClass("navbar-social-btn-width-with-no-text");
            };

            $(value).addClass("navbar-social-btn-width-with-text");
          });

        // For medium or smaller devices
        } else {

          // Remove text from social media buttons
          $.each(social_navbar_text, function(index, value) {
            $(value).text("");
          });

          // Apply css changes for social media buttons without text
          $.each(navbar_social_links, function(index, value){
            if ($(value).hasClass("navbar-social-btn-width-with-text")) {
              $(value).removeClass("navbar-social-btn-width-with-text");
            };

            $(value).addClass("navbar-social-btn-width-with-no-text");
          });
        };          
      });
    }); 

    //  If window is resized to a larger width with the navbar dropdown active, de-activate navbar dropdown
    $(function(){
      $(window).on('resize', function(){
        if($(this).width() > 991.98 && dropDownActive == true) {
          $("#navbar-mobile-btn").click();
        };
      });
    });


    $('.navbar-dropdown-toggle-button').on('click', function () {
      $('.navbar-dropdown-animation').toggleClass('open');
    });

    $(function () {
      $('[data-toggle="tooltip"]').tooltip();
    });
  });


  