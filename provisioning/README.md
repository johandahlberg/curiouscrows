Setup and deployment instructions
=================================

Preparations
------------

To get things going you need to have Ansible installed, and you need to ensure that 
you have the AWS-related environment variables set.

    AWSAccessKeyId=<your WSAccessKeyId>
    AWSSecretKey=<your AWSSecretKey>

You also need to install the boto library for Ansible to manage a dynamic ec2 based inventory.

    pip install boto
    
Start a server
--------------
    
    ansible-playbook --private-key=<key_path> setup_server.yml

Deploy the app
--------------
    
    ansible-playbook -i ec2.py --private-key=<key_path> deploy_app.yml


Teardown
--------

Remove the server once you're done:

    ansible-playbook --private-key=<key_path> setup_server.yml --extra-vars="server_count=0"